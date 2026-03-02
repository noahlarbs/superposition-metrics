import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import einops
import argparse
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# FAKE QUANTIZATION OPS (STE) 
# ---------------------------------------------------------
def quantize_w8(w):
    scale = 127.0 / w.abs().max().clamp(min=1e-8)
    w_q = torch.round(w * scale).clamp(-128, 127) / scale
    return w + (w_q - w).detach()

def quantize_a16(x):
    scale = 32767.0 / x.abs().max().clamp(min=1e-8)
    x_q = torch.round(x * scale).clamp(-32768, 32767) / scale
    return x + (x_q - x).detach()

def quantize_w4(w):
    scale = 7.0 / w.abs().max().clamp(min=1e-8)
    w_q = torch.round(w * scale).clamp(-8, 7) / scale
    return w + (w_q - w).detach()

def quantize_a8(x):
    return quantize_w8(x)

def quantize_ternary(w):
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-1, 1) * scale
    return w + (w_q - w).detach()

def quantize_q2_k(w):
    scale = 1.0 / w.abs().max().clamp(min=1e-8)
    w_q = torch.round(w * scale).clamp(-1, 1) / scale 
    return w + (w_q - w).detach()

def quantize_iq2_xxs(w):
    scale = w.abs().max().clamp(min=1e-8)
    w_norm = w / scale
    thresholds = torch.tensor([-1.0, -0.33, 0.33, 1.0], device=w.device)
    diffs = torch.abs(w_norm.unsqueeze(-1) - thresholds)
    min_idx = torch.argmin(diffs, dim=-1)
    w_q = thresholds[min_idx] * scale
    return w + (w_q - w).detach()

# ---------------------------------------------------------
# PRECISION DECAY AGES
# ---------------------------------------------------------
def apply_precision_age(w, age, base_quant_fn):
    if age == 0: 
        return w # FP32
    elif age == 1:
        return w.half().float()  
    elif age == 2:
        return quantize_w8(w)
    else:
        return base_quant_fn(w)

# ---------------------------------------------------------
# PROGRESSIVE TOY MODEL OVERRIDES
# ---------------------------------------------------------
class ProgressiveToyModel(nn.Module):
    def __init__(self, n_features, n_hidden, importance, feature_probability, base_quant='W8A16', is_ageing=True):
        super().__init__()
        self.n = n_features
        self.m = n_hidden
        self.base_quant = base_quant
        self.is_ageing = is_ageing
        
        self.W = nn.Parameter(torch.randn(n_features, n_hidden) / math.sqrt(n_hidden))
        self.b = nn.Parameter(torch.zeros(n_features))
        
        self.importance = importance
        self.feature_probability = feature_probability
        self.neuron_ages = torch.full((n_hidden,), 3.0, dtype=torch.float32)

    def _get_base_quant_fn(self):
        if self.base_quant == 'W8A16': return quantize_w8
        if self.base_quant == 'W4A8': return quantize_w4
        if self.base_quant == 'Ternary': return quantize_ternary
        if self.base_quant == 'q2_k': return quantize_q2_k
        if self.base_quant == 'iq2_xxs': return quantize_iq2_xxs
        return lambda x: x

    def forward(self, features):
        q_fn = self._get_base_quant_fn()
        W_sim = torch.empty_like(self.W)
        for i in range(self.m):
            age = int(self.neuron_ages[i].item())
            if self.is_ageing:
                W_sim[:, i] = apply_precision_age(self.W[:, i], age, q_fn)
            else:
                if age >= 3:
                     W_sim[:, i] = q_fn(self.W[:, i])
                else:
                     W_sim[:, i] = self.W[:, i]

        if self.base_quant in ['W8A16']:
            x_sim = quantize_a16(features)
        elif self.base_quant in ['W4A8', 'Ternary', 'q2_k', 'iq2_xxs']:
             x_sim = quantize_a8(features)
        else:
             x_sim = features

        hidden = x_sim @ W_sim
        out = F.relu(hidden @ W_sim.T + self.b)
        return out, hidden

    def generate_batch(self, n_batch):
        feat = torch.rand((n_batch, self.n), device=self.W.device)
        batch = torch.where(
            torch.rand((n_batch, self.n), device=self.W.device) <= self.feature_probability,
            feat,
            torch.zeros((), device=self.W.device),
        )
        return batch

def compute_effective_rank(W):
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    p = (S + 1e-12) / (S + 1e-12).sum()
    entropy = -torch.sum(p * torch.log(p))
    effective_rank = torch.exp(entropy)
    return effective_rank.item()

def split_polysemantic_neurons(model):
    device = model.W.device
    W_old = model.W.detach()
    n, m = W_old.shape
    
    W_norm = W_old / (1e-5 + torch.linalg.norm(W_old, 2, dim=0, keepdim=True))
    interference = torch.abs(W_norm.T @ W_norm)
    interference.fill_diagonal_(0)
    
    poly_scores = interference.sum(dim=0)
    threshold = torch.median(poly_scores)
    to_expand = (poly_scores >= threshold).nonzero(as_tuple=True)[0]
    
    n_new = len(to_expand)
    if n_new == 0:
        return False
        
    m_new = m + n_new
    new_ages = model.neuron_ages + 1.0
    spawned_ages = torch.zeros(n_new, dtype=torch.float32)
    model.neuron_ages = torch.cat([new_ages, spawned_ages])
    
    W_spawn = torch.zeros(n, n_new, device=device)
    for idx, parent_col in enumerate(to_expand):
        w = W_old[:, parent_col]
        conn_idx = (w.abs() > 1e-5).nonzero(as_tuple=True)[0]
        if len(conn_idx) > 1:
            mid = len(conn_idx) // 2
            child_conns = conn_idx[mid:]
            parent_conns = conn_idx[:mid]
            W_spawn[:, idx][child_conns] = w[child_conns]
            W_old[:, parent_col][child_conns] = 0.0 
    
    new_W = torch.cat([W_old, W_spawn], dim=1)
    
    # We must reset the optimizer state when parameters change! 
    # Returning the new parameters directly so the training loop can rebuild the optimizer.
    return True, new_W, m_new

def get_lr(step, lr, n_steps, warmup_steps=1000):
        step = step + 1
        min_lr = 0.05 * lr
        if warmup_steps < n_steps:
            if step < warmup_steps:
                return lr * step / warmup_steps
            else:
                return (lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * (step - warmup_steps) / (n_steps - warmup_steps))
                ) + min_lr
        else:
            return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / n_steps)) + min_lr

def run_experiment(base_quant, is_ageing, args, device):
    print(f"\n=======================================================")
    print(f"RUNNING: Quant={base_quant} | FPE Regime={'Progressive Ageing' if is_ageing else 'Fixed FP32'}")
    print(f"=======================================================")
    
    n_features = args.n_features
    # Start tight
    m_start = args.m_start
    n_steps = args.n_steps
    patience = args.patience
    tolerance = args.tolerance
    
    importance = (args.importance_base ** -torch.linspace(0, 1, n_features))
    feature_probability = (args.prob_base ** -torch.linspace(0, 1, n_features))
    importance = importance.to(device)
    feature_probability = feature_probability.to(device)

    model = ProgressiveToyModel(
        n_features, 
        m_start, 
        importance, 
        feature_probability, 
        base_quant=base_quant, 
        is_ageing=is_ageing
    ).to(device)
    
    # Base LR for this run
    base_lr = 5e-3 if base_quant == 'Ternary' else 1e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)
    
    losses = []
    eff_ranks = []
    fpe_events = []
    
    plateau_counter = 0
    best_eff_rank = 0.0
    
    for step in range(n_steps):
        batch = model.generate_batch(args.batch_size)

        lr = get_lr(step, base_lr, n_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        out, hidden = model(batch)
        error = (model.importance * (batch.abs() - out)**2)
        loss = error.mean(dim=0).sum()
        loss.backward()
        
        # Enforce sparsity structure preservation if we've split
        # Any zeroed weight during splitting should remain exactly zeroed
        sparsity_mask = (model.W.data.abs() > 0).float()
        model.W.grad.data *= sparsity_mask
        optimizer.step()
        
        if step % args.log_interval == 0:
            with torch.no_grad():
                W_norm = model.W / (1e-5 + torch.linalg.norm(model.W, 2, dim=0, keepdim=True))
                # compute true D_pr alternative over batch
                erank = compute_effective_rank(W_norm.T @ W_norm)
                
            losses.append(loss.item())
            eff_ranks.append(erank)
            
            # Check plateau
            if erank > best_eff_rank + tolerance:
                best_eff_rank = erank
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            # TRIGGER FPE DETONATION!
            if plateau_counter >= patience and model.m < args.m_max:
                print(f"  [Step {step}] 🎯 Triggering FPE! EffRank plateaued at {erank:.3f}")
                res = split_polysemantic_neurons(model)
                if res != False:
                    _, new_W, new_m = res
                    model.W = nn.Parameter(new_W)
                    model.m = new_m
                    
                    # Rebuild optimizer
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
                    best_eff_rank = compute_effective_rank(new_W.T @ new_W)
                    plateau_counter = 0
                    fpe_events.append({'step': step, 'm': new_m, 'eff_rank': best_eff_rank})
                    print(f"  --> Detonated to {new_m} widths! New eff rank {best_eff_rank:.3f}")
                    
            if step % (args.log_interval * 10) == 0:
                print(f"  Step {step}/{n_steps} | Loss {loss.item():.4f} | E-Rank {erank:.2f} | Width {model.m}")
                
    return {
        'losses': losses,
        'eff_ranks': eff_ranks,
        'fpe_events': fpe_events,
        'final_loss': losses[-1] if len(losses)>0 else 0,
        'final_m': model.m
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_features", type=int, default=100)
    parser.add_argument("--m_start", type=int, default=10)
    parser.add_argument("--m_max", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10, help="Check intervals before trigger")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Eff Rank delta tolerance")
    
    parser.add_argument("--importance_base", type=float, default=100)
    parser.add_argument("--prob_base", type=float, default=20)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    quants = ['W8A16', 'W4A8', 'Ternary', 'q2_k', 'iq2_xxs']
    regimes = [{'name': 'Ageing', 'is_ageing': True}, {'name': 'Fixed', 'is_ageing': False}]
    
    results = {}
    for q in quants:
        for r in regimes:
            k = f"{q}_{r['name']}"
            results[k] = run_experiment(q, r['is_ageing'], args, device)
            
    # Compile Graph
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle("Progressive Quantization Matrix (Toy Model of Superposition)", fontsize=16)
    
    for c_idx, q in enumerate(quants):
        ax_loss = axes[0, c_idx]
        ax_erank = axes[1, c_idx]
        
        ax_loss.set_title(f"{q} Base Quant Constraint")
        ax_loss.set_ylabel("Reconstruction Loss")
        ax_loss.set_yscale('log')
        
        ax_erank.set_ylabel("Effective Rank")
        ax_erank.set_xlabel("Steps (x50)")
        
        for r in regimes:
            k = f"{q}_{r['name']}"
            res = results[k]
            color = 'blue' if r['is_ageing'] else 'orange'
            
            ax_loss.plot(res['losses'], label=f"{r['name']}", color=color, alpha=0.8)
            ax_erank.plot(res['eff_ranks'], label=f"{r['name']}", color=color, alpha=0.8)
            
            # Plot Detonation events
            for event in res['fpe_events']:
                step_idx = event['step'] // args.log_interval
                ax_loss.axvline(x=step_idx, color=color, linestyle='--', alpha=0.3)
                ax_erank.axvline(x=step_idx, color=color, linestyle='--', alpha=0.3)

        ax_loss.legend()
        ax_erank.legend()

    plt.tight_layout()
    output_path = "../outputs/progressive_quant_matrix.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Matrix graph saved to {output_path}")

if __name__ == "__main__":
    main()

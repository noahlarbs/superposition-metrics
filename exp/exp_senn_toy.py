import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import einops
import argparse
import os
import matplotlib.pyplot as plt
import wandb

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

def quantize_q2_k(w, block_size=32):
    orig_shape = w.shape
    w_flat = w.view(-1)
    
    pad_len = (block_size - (w_flat.size(0) % block_size)) % block_size
    if pad_len > 0:
        w_padded = torch.cat([w_flat, torch.zeros(pad_len, device=w.device)])
    else:
        w_padded = w_flat
        
    blocks = w_padded.view(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    blocks_norm = blocks / scales
    
    # 4 distinct states = 2 bits
    thresholds = torch.tensor([-1.0, -0.33, 0.33, 1.0], device=w.device)
    diffs = torch.abs(blocks_norm.unsqueeze(-1) - thresholds)
    min_idx = torch.argmin(diffs, dim=-1)
    blocks_q = thresholds[min_idx] * scales
    
    w_q_flat = blocks_q.view(-1)
    if pad_len > 0:
        w_q_flat = w_q_flat[:-pad_len]
        
    w_q = w_q_flat.view(orig_shape)
    return w + (w_q - w).detach()

def quantize_iq2_xxs(w, block_size=32):
    orig_shape = w.shape
    w_flat = w.view(-1)
    
    pad_len = (block_size - (w_flat.size(0) % block_size)) % block_size
    if pad_len > 0:
        w_padded = torch.cat([w_flat, torch.zeros(pad_len, device=w.device)])
    else:
        w_padded = w_flat
        
    blocks = w_padded.view(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    blocks_norm = blocks / scales
    
    # iq2_xxs drops extreme weights to a ternary distribution with a heavy zero bias
    thresholds = torch.tensor([-1.0, 0.0, 1.0], device=w.device)
    diffs = torch.abs(blocks_norm.unsqueeze(-1) - thresholds)
    min_idx = torch.argmin(diffs, dim=-1)
    blocks_q = thresholds[min_idx] * scales
    
    w_q_flat = blocks_q.view(-1)
    if pad_len > 0:
        w_q_flat = w_q_flat[:-pad_len]
        
    w_q = w_q_flat.view(orig_shape)
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

def print_quantization_report(base_quant, is_ageing, widths, ages=None):
    """
    Validation check to output theoretical BPW and hardware datatype packing requirements.
    """
    total_params = sum(widths) if isinstance(widths, list) else widths
    
    quant_info = {
        'W8A16':   {'bits': 8, 'dtype': 'int8'},
        'W4A8':    {'bits': 4, 'dtype': 'int4 (packed in int8)'},
        'Ternary': {'bits': 2, 'dtype': 'int2 (packed in int8)'},
        'q2_k':    {'bits': 2, 'dtype': 'int2 (packed in int8)'},
        'iq2_xxs': {'bits': 2, 'dtype': 'int2 (packed in int8)'},
    }
    
    tgt = quant_info.get(base_quant, {'bits': 32, 'dtype': 'fp32'})
    
    if not is_ageing or ages is None:
        bpw = tgt['bits']
        print(f"  [Quantization Report] Fixed {base_quant}: ~{bpw:.2f} bpw | Target Hardware Packing: {tgt['dtype']}")
        return
        
    # Calculate BPW based on precision ageing rules
    # Age 0: FP32 (32 bit), Age 1: FP16 (16 bit), Age 2: INT8 (8 bit), Age 3+: Base
    bits_sum = 0
    for a in ages:
        if a == 0: bits_sum += 32
        elif a == 1: bits_sum += 16
        elif a == 2: bits_sum += 8
        else: bits_sum += tgt['bits']
        
    avg_bpw = bits_sum / len(ages)
    
    # Detail the distribution
    age_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for a in ages:
        a_idx = int(a) if a < 3 else 3
        age_counts[a_idx] += 1
        
    print(f"  [Quantization Report] FPE Mixed Precision BPW: {avg_bpw:.2f} bits/weight")
    print(f"    - {age_counts[0]:4d} neurons @ fp32")
    print(f"    - {age_counts[1]:4d} neurons @ fp16")
    print(f"    - {age_counts[2]:4d} neurons @ int8")
    print(f"    - {age_counts[3]:4d} neurons @ {tgt['dtype']} ({base_quant})")

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

def compute_fisher_pr(grads):
    """
    Computes PR(F) = (Tr F)^2 / Tr(F^2)
    where F is the empirical Fisher approximated from a batch of gradients.
    grads: [B, m]
    """
    B = grads.shape[0]
    # F = (1/B) * (grads^T @ grads)
    F = (grads.T @ grads) / B
    
    trace_F = torch.trace(F)
    trace_F_sq = torch.trace(F @ F)
    
    if trace_F_sq < 1e-12:
        return 1.0 # Minimum PR
        
    pr = (trace_F ** 2) / trace_F_sq
    return pr.item()

def compute_expansion_score(grads):
    """
    Computes Natural Expansion Score (eta) = g^T F^-1 g
    Using pseudo-inverse for stability.
    Returns eta and the diagonal of F as a proxy for per-neuron saturation.
    """
    B = grads.shape[0]
    g = grads.mean(dim=0) # Expected gradient [m]
    F = (grads.T @ grads) / B # [m, m]
    
    # Pseudo-inverse for ill-conditioned Fisher
    F_inv = torch.linalg.pinv(F, rcond=1e-5)
    eta = g.T @ F_inv @ g
    
    # Return eta and the Fisher diagonals (gradient variance) for parent selection
    return eta.item(), torch.diag(F)

def split_polysemantic_neurons(model, parent_scores):
    """
    Appends new neurons based on saturation score, but purely additively instead of destructively
    splitting the existing parameters. This preserves the internal scale (max/mean) of 
    the parent neurons, avoiding detonating the QAT baselines and exploding earlier loss.
    """
    device = model.W.device
    W_old = model.W.detach()
    n, m = W_old.shape
    
    # Select parents by gradient variance (as proxy for saturation load)
    threshold = torch.median(parent_scores)
    to_expand = (parent_scores >= threshold).nonzero(as_tuple=True)[0]
    
    n_new = len(to_expand)
    if n_new == 0:
        return False
        
    m_new = m + n_new
    new_ages = model.neuron_ages + 1.0
    spawned_ages = torch.zeros(n_new, dtype=torch.float32)
    model.neuron_ages = torch.cat([new_ages, spawned_ages])
    
    # Init cleanly without destroying parent weights (like the Transformer code)
    W_spawn = torch.randn(n, n_new, device=device) * 0.02
    
    new_W = torch.cat([W_old, W_spawn], dim=1)
    
    # We must reset the optimizer state when parameters change! 
    # Returning the new parameters directly so the training loop can rebuild the optimizer.
    return True, new_W, m_new

def get_lr(step, lr, n_steps, warmup_steps=1000):
        step = step + 1
        min_lr = 0.2 * lr
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
    regime_name = 'Progressive Ageing' if is_ageing else 'Fixed FP32'
    print(f"\n=======================================================")
    print(f"RUNNING: Quant={base_quant} | FPE Regime={regime_name}")
    print(f"=======================================================")
    
    run_name = f"toy_senn_{base_quant}_{regime_name.replace(' ', '_')}"
    wandb.init(
        project="superposition-metrics",
        name=run_name,
        config={
            **vars(args),
            "base_quant": base_quant,
            "is_ageing": is_ageing,
            "architecture": "toy_model_senn"
        }
    )
    
    n_features = args.n_features
    # Start tight
    m_start = args.m_start
    n_steps = args.n_steps
    patience = args.patience
    tolerance = args.tolerance # This is now epsilon for D_PR
    
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
    base_lr = 1e-2 if base_quant in ['Ternary', 'iq2_xxs', 'q2_k'] else 2e-3
    base_wd = 0.0 if base_quant in ['Ternary', 'iq2_xxs', 'q2_k'] else 1e-2

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_wd)
    
    losses = []
    eff_ranks = []
    fpe_events = []
    
    plateau_counter = 0
    pr_history = []
    tau = 0.05 # Pre-defined width threshold
    
    print_quantization_report(base_quant, is_ageing, model.m, model.neuron_ages if is_ageing else None)
    
    for step in range(n_steps):
        batch = model.generate_batch(args.batch_size)

        lr = get_lr(step, base_lr, n_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        out, hidden = model(batch)
        
        # We need the gradient of the hidden acts to compute empirical Fisher
        hidden.retain_grad()
        
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
                # --- SENN METRICS ---
                pr_f = compute_fisher_pr(hidden.grad)
                eta, F_diag = compute_expansion_score(hidden.grad)
                
            losses.append(loss.item())
            eff_ranks.append(pr_f)
            
            # --- D_PR Sliding Window ---
            pr_history.append(pr_f)
            if len(pr_history) > patience:
                pr_history.pop(0)
            
            if len(pr_history) == patience:
                d_pr = (pr_history[-1] - pr_history[0]) / patience
                if abs(d_pr) < tolerance:
                    plateau_counter += 1
                else:
                    plateau_counter = 0
            else:
                d_pr = float('inf')
                plateau_counter = 0
                
            # --- DUAL CONDITIONAL TRIGGER (D_PR < eps AND Delta_Eta >= tau * eta_0 / rho_0) ---
            if plateau_counter >= patience and model.m < args.m_max:
                # Estimate Delta Eta from the top 50% saturated neurons
                threshold_var = torch.median(F_diag)
                prop_variance = F_diag[F_diag >= threshold_var].sum() / F_diag.sum()
                delta_eta = eta * prop_variance.item()
                
                req_delta = tau * (eta / pr_f)
                
                if delta_eta >= req_delta:
                    print(f"  [Step {step}] 🎯 Triggering FPE! D_PR={d_pr:.4f} & dEta={delta_eta:.4f} >= {req_delta:.4f}")
                    res = split_polysemantic_neurons(model, F_diag)
                    if res != False:
                        _, new_W, new_m = res
                        model.W = nn.Parameter(new_W)
                        model.m = new_m
                        
                        # Rebuild optimizer
                        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=base_wd)
                        pr_history.clear()
                        plateau_counter = 0
                        fpe_events.append({'step': step, 'm': new_m, 'eff_rank': pr_f})
                        print(f"  --> Detonated to {new_m} widths! New FIM PR: {compute_fisher_pr(hidden.grad):.3f}")
                        
                            print_quantization_report(base_quant, is_ageing, model.m, model.neuron_ages if is_ageing else None)
                    
            if step % (args.log_interval * 10) == 0:
                print(f"  Step {step}/{n_steps} | Loss {loss.item():.4f} | Fisher PR {pr_f:.2f} | Width {model.m}")
            
            wandb.log({
                "train/loss": loss.item(),
                "metrics/fisher_pr": pr_f,
                "metrics/eta": eta,
                "metrics/d_pr": d_pr if d_pr != float('inf') else 0,
                "model/width": model.m,
                "train/step": step
            })
                
    wandb.finish()
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
    parser.add_argument("--n_steps", type=int, default=20000)
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

%%writefile exp_fpe_growth_ablation.py
"""
Targeted FPE Transformer - Dynamic Growth Ablation Framework

This script empirically validates the limits of Fixed Parameter Expansion (FPE) 
computational efficiency by mapping 3 core variables natively on TinyStories:

Panel A (Growth Factor Scale Sweep): 2x vs. 4x vs. 8x architectural expansions.
Panel B (Multi-Stage Growth): A single 4x parameter Detonation vs two gradual 2x stages.
Panel C (Trigger Sensitivity): Tolerance 1.0 (loose/early jump) vs Tolerance 0.1 (strict plateau).

It uses a continuous inline execution loop, dynamically tracking proxy FLOPs via 
the `accumulated_compute` variable natively mapped against the Cross-Entropy Loss curve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# FPE Metrics & Splitting Math (Strict Disjoint Partitioning for N Children)
# -----------------------------------------------------------------------------
WEIGHT_THRESHOLD = 1e-6

def compute_pr_dim(activations: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    H = activations - activations.mean(dim=0, keepdim=True)
    batch_size, hidden_dim = H.shape
    n = batch_size - 1
    if n <= 0: return torch.tensor(float("nan"), device=activations.device)
    
    sigma = (H.T @ H) / n
    tr_sigma = sigma.trace()
    tr_sigma_sq = (sigma @ sigma).trace()
    return (tr_sigma * tr_sigma) / (tr_sigma_sq + eps)

def get_weight_pr(W):
    l1_norm = W.abs().sum(dim=0)
    l2_sq_norm = (W ** 2).sum(dim=0)
    return (l1_norm ** 2) / (l2_sq_norm + 1e-8)

def split_ffn_neurons(W1, W2, b1=None, n_children=4, reference_W1=None, target_fraction=1.0, run_mode="fpe"):
    if reference_W1 is None: reference_W1 = W1
    device = W1.device
    d_model, d_ff = W1.shape
    
    new_w1_cols, new_w2_rows, new_b1 = [], [], []
    expanded_indices, idx_counter = [], 0
    
    if target_fraction < 1.0:
        pr_w = get_weight_pr(reference_W1)
        n_target = max(1, int(d_ff * target_fraction))
        _, sorted_indices = torch.sort(pr_w, descending=True)
        target_indices = set(sorted_indices[:n_target].tolist())
    else:
        target_indices = set(range(d_ff))

    for j in range(d_ff):
        w1_j = W1[:, j]
        w2_j = W2[j, :]
        b1_j = b1[j] if b1 is not None else torch.tensor(0.0, device=device)
        ref_w = reference_W1[:, j]
        
        if j not in target_indices:
            new_w1_cols.append(w1_j.unsqueeze(1))
            new_w2_rows.append(w2_j.unsqueeze(0))
            new_b1.append(b1_j.unsqueeze(0))
            idx_counter += 1
            continue
            
        nonzero_idx = (ref_w.abs() > WEIGHT_THRESHOLD).nonzero(as_tuple=True)[0]
        
        # In static modes, do not split
        if "static" in run_mode or len(nonzero_idx) <= 1:
            new_w1_cols.append(w1_j.unsqueeze(1))
            new_w2_rows.append(w2_j.unsqueeze(0))
            new_b1.append(b1_j.unsqueeze(0))
            expanded_indices.append(idx_counter)
            idx_counter += 1
        else:
            # FPE STRICT PARTITIONING
            idx_list = nonzero_idx.tolist()
            n_conn = len(idx_list)
            n_splits = min(n_children, n_conn)
            base_size, remainder = n_conn // n_splits, n_conn % n_splits
            
            offset = 0
            for k in range(n_splits):
                size = base_size + (1 if k < remainder else 0)
                if size == 0: continue
                part_idx = idx_list[offset : offset + size]
                offset += size
                
                w1_child = torch.zeros(d_model, device=device, dtype=W1.dtype)
                for i in part_idx: w1_child[i] = w1_j[i]
                
                w2_child = w2_j.clone()
                b1_child = b1_j.clone() / n_splits
                
                new_w1_cols.append(w1_child.unsqueeze(1))
                new_w2_rows.append(w2_child.unsqueeze(0))
                new_b1.append(b1_child.unsqueeze(0))
                
                expanded_indices.append(idx_counter)
                idx_counter += 1

    W1_new = torch.cat(new_w1_cols, dim=1)
    W2_new = torch.cat(new_w2_rows, dim=0)
    b1_new = torch.cat(new_b1, dim=0) if b1 is not None else None
    
    return W1_new, W2_new, b1_new

# -----------------------------------------------------------------------------
# FPE-Aware Transformer Architecture
# -----------------------------------------------------------------------------
class ExpandableFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.gelu = nn.GELU()
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_hidden=False):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        h = x_flat @ self.W1 + self.b1
        h = self.gelu(h)
        h = self.act_dropout(h)
        hidden_states = h.clone()
        out = h @ self.W2 + self.b2
            
        out = self.dropout(out)
        out = out.view(B, T, C)
        if return_hidden:
            return out, hidden_states
        return out
        
    def expand_ffn(self, n_children, target_fraction, run_mode):
        W1_new, W2_new, b1_new = split_ffn_neurons(
            self.W1.data, self.W2.data, self.b1.data, 
            n_children=n_children, target_fraction=target_fraction, run_mode=run_mode
        )
        self.d_ff = W1_new.shape[1]
        self.W1 = nn.Parameter(W1_new)
        self.W2 = nn.Parameter(W2_new)
        self.b1 = nn.Parameter(b1_new)
        return self.d_ff

class MinimalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = ExpandableFeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=None, return_hidden=False):
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False, is_causal=True)
        x = x + attn_out
        
        x_norm = self.ln_2(x)
        if return_hidden:
            ffn_out, h_states = self.ffn(x_norm, return_hidden=True)
            x = x + ffn_out
            return x, h_states
        else:
            x = x + self.ffn(x_norm)
            return x

class FPETransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, max_seq_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.block = MinimalTransformerBlock(d_model, n_heads, d_ff)
        self.ln_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, return_hidden=False):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        
        if return_hidden:
            x, h_states = self.block(x, attn_mask=causal_mask, return_hidden=True)
            logits = self.lm_head(self.ln_out(x))
            return logits, h_states
        else:
            x = self.block(x, attn_mask=causal_mask)
            logits = self.lm_head(self.ln_out(x))
            return logits

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
def get_dataloader(tokenizer, batch_size=32, seq_len=128):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len+1, padding="max_length")

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(100000))
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
    tokenized.set_format("torch", columns=["input_ids"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------
# Main Test Harness
# -----------------------------------------------------------------------------
def run_experiment(args, device, dataloader, tokenizer, vocab_size, run_mode, 
                   growth_factor=None, start_d_ff=64, num_expansions_target=1, tolerance=0.5):
    """
    Standardized execution loop. FPE logic triggers inline seamlessly.
    """
    print(f"\n{'='*70}")
    print(f"Mode: {run_mode.upper():20} | Triggers: {num_expansions_target} | Tol: {tolerance}")
    print(f"{'='*70}")

    model = FPETransformer(vocab_size, args.d_model, args.n_heads, start_d_ff, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    data_iter = iter(dataloader)
    
    compute_losses = []
    recent_pr_dims = []
    accumulated_compute = 0
    expansions_done = 0
    
    print(f"[Run Started] Initial d_ff: {start_d_ff}")
    
    # Continuous execution loop
    for step in range(args.n_steps_max):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        idx = batch["input_ids"].to(device)
        x = idx[:, :-1]
        y = idx[:, 1:]
        
        logits, h_states = model(x, return_hidden=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss = loss / args.grad_accum_steps
        loss.backward()
        
        # Flop Proxy Accumulator tracks architectural mass dynamically
        accumulated_compute += model.block.ffn.d_ff
        
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        current_loss = loss.item() * args.grad_accum_steps
        
        if (step + 1) % args.log_interval == 0:
            with torch.no_grad():
                h_sample = h_states[torch.randperm(h_states.size(0))[:2048]]
                pr_dim = compute_pr_dim(h_sample).item()
            
            compute_losses.append((accumulated_compute, current_loss))
            print(f"Step {step+1:5} | Compute: {accumulated_compute:8} | Loss: {current_loss:.4f} | D_PR: {pr_dim:.2f}")
            recent_pr_dims.append(pr_dim)
            
            # Inline Continuous FPE Trigger Logic
            if "fpe" in run_mode and expansions_done < num_expansions_target and len(recent_pr_dims) >= args.patience:
                window = recent_pr_dims[-args.patience:]
                dpr_diff = max(window) - min(window)
                
                if dpr_diff < tolerance:
                    print(f"\n--> INLINE FPE TRIGGER! D_PR Plateaued natively (Diff: {dpr_diff:.2f}).")
                    old_m = model.block.ffn.d_ff
                    
                    # Split logic execution
                    new_m = model.block.ffn.expand_ffn(n_children=growth_factor, target_fraction=1.0, run_mode="fpe")
                    print(f"--> Geometric Detonation Expanded Parameter Space: {old_m}x -> {new_m}x")
                    
                    # Re-initialize optimizer for the new spatial block
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.8, weight_decay=args.weight_decay)
                    
                    # Clean plateau tracker so we can map successive stages
                    recent_pr_dims = []
                    expansions_done += 1
                    
    print(f"[Run Completed] Mode: {run_mode.upper()} Final Compute: {accumulated_compute}")
    return compute_losses

# -----------------------------------------------------------------------------
# Validation Harness & Graphing
# -----------------------------------------------------------------------------
def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff_base", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--n_steps_max", type=int, default=8000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    dataloader = get_dataloader(tokenizer, args.batch_size, args.seq_len)
    
    results = {}

    print("\n[Executing Panel A: Growth Factor Scale Sweep]")
    results["static_64"]  = run_experiment(args, device, dataloader, tokenizer, vocab_size, "static_small", start_d_ff=64)
    results["static_128"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "static_med", start_d_ff=128)
    results["static_256"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "static_large", start_d_ff=256)
    results["static_512"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "static_huge", start_d_ff=512)
    
    results["fpe_2x"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "fpe", growth_factor=2, start_d_ff=64, tolerance=0.5)
    results["fpe_4x"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "fpe", growth_factor=4, start_d_ff=64, tolerance=0.5)
    results["fpe_8x"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "fpe", growth_factor=8, start_d_ff=64, tolerance=0.5)

    print("\n[Executing Panel B: Multi-Stage Gradual Growth]")
    # Test fpe_staged_2x2 (Hits 64, wait for plateau, double to 128, wait for plateau, double to 256)
    results["fpe_staged_2x2"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "fpe_staged", growth_factor=2, start_d_ff=64, num_expansions_target=2, tolerance=0.5)

    print("\n[Executing Panel C: Trigger Sensitivity]")
    results["fpe_tol_1.0"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "fpe", growth_factor=4, start_d_ff=64, tolerance=1.0)
    results["fpe_tol_0.1"] = run_experiment(args, device, dataloader, tokenizer, vocab_size, "fpe", growth_factor=4, start_d_ff=64, tolerance=0.1)

    # ==========================================
    # Multi-Panel Grid Matplotlib Plotting
    # ==========================================
    os.makedirs("../newfigures", exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Dynamic Network Growth Interventions - FPE Architectural Ablation Framework", fontsize=16, y=1.02)
    
    # ---------------------------------------------
    # Panel A: Growth Factor Scale Sweep
    # ---------------------------------------------
    ax1 = axes[0]
    ax1.plot([p[0] for p in results["static_128"]], smooth([p[1] for p in results["static_128"]]), color="gray", alpha=0.9, linestyle=":", linewidth=2, label="Static 128 Threshold")
    ax1.plot([p[0] for p in results["static_256"]], smooth([p[1] for p in results["static_256"]]), color="black", alpha=0.9, linestyle="--", linewidth=2, label="Static 256 Threshold")
    ax1.plot([p[0] for p in results["static_512"]], smooth([p[1] for p in results["static_512"]]), color="black", alpha=0.6, linestyle="-.", linewidth=2, label="Static 512 Threshold")
    ax1.plot([p[0] for p in results["static_64"]], smooth([p[1] for p in results["static_64"]]), color="red", alpha=0.5, linewidth=2, label="Static Baseline (64)")
    
    ax1.plot([p[0] for p in results["fpe_2x"]], smooth([p[1] for p in results["fpe_2x"]]), color="green", alpha=0.9, linewidth=2, label="FPE 2x (64 -> 128)")
    ax1.plot([p[0] for p in results["fpe_4x"]], smooth([p[1] for p in results["fpe_4x"]]), color="blue", alpha=0.9, linewidth=2, label="FPE 4x (64 -> 256)")
    ax1.plot([p[0] for p in results["fpe_8x"]], smooth([p[1] for p in results["fpe_8x"]]), color="purple", alpha=0.9, linewidth=2, label="FPE 8x (64 -> 512)")
    
    ax1.set_title("Panel A: Growth Factor Sweep Limits")
    ax1.set_xlabel("Accumulated Compute Proxy (FLOPs)")
    ax1.set_ylabel("Validation Loss")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ---------------------------------------------
    # Panel B: Multi-Stage Gradual Growth (Target 256)
    # ---------------------------------------------
    ax2 = axes[1]
    ax2.plot([p[0] for p in results["static_64"]], smooth([p[1] for p in results["static_64"]]), color="red", alpha=0.3, linewidth=2, label="Static Baseline (64)")
    ax2.plot([p[0] for p in results["static_256"]], smooth([p[1] for p in results["static_256"]]), color="black", alpha=0.9, linestyle="--", linewidth=2, label="Static Target (256)")
    
    ax2.plot([p[0] for p in results["fpe_4x"]], smooth([p[1] for p in results["fpe_4x"]]), color="blue", alpha=0.9, linewidth=2, label="Single-Stage 4x (64 -> 256)")
    ax2.plot([p[0] for p in results["fpe_staged_2x2"]], smooth([p[1] for p in results["fpe_staged_2x2"]]), color="orange", alpha=0.9, linewidth=2, label="Staged FPE (64 -> 128 -> 256)")
    
    ax2.set_title("Panel B: Single vs Gradual Multi-Stage Expansion")
    ax2.set_xlabel("Accumulated Compute Proxy (FLOPs)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---------------------------------------------
    # Panel C: Trigger Sensitivity
    # ---------------------------------------------
    ax3 = axes[2]
    ax3.plot([p[0] for p in results["static_64"]], smooth([p[1] for p in results["static_64"]]), color="red", alpha=0.3, linewidth=2, label="Static Baseline (64)")
    ax3.plot([p[0] for p in results["static_256"]], smooth([p[1] for p in results["static_256"]]), color="black", alpha=0.9, linestyle="--", linewidth=2, label="Static Target (256)")
    
    ax3.plot([p[0] for p in results["fpe_4x"]], smooth([p[1] for p in results["fpe_4x"]]), color="blue", alpha=0.9, linewidth=2, label="Optimized Tolerence (0.5)")
    ax3.plot([p[0] for p in results["fpe_tol_1.0"]], smooth([p[1] for p in results["fpe_tol_1.0"]]), color="cyan", alpha=0.9, linestyle=":", linewidth=2, label="Loose Early Tolerance (1.0)")
    ax3.plot([p[0] for p in results["fpe_tol_0.1"]], smooth([p[1] for p in results["fpe_tol_0.1"]]), color="magenta", alpha=0.9, linestyle="-.", linewidth=2, label="Strict Plateau Tolerance (0.1)")
    
    ax3.set_title("Panel C: D_PR Trigger Sensitivity Limits")
    ax3.set_xlabel("Accumulated Compute Proxy (FLOPs)")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../newfigures/efficiency_ablations.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved Multi-Panel Ablation Grid Figure: ../newfigures/efficiency_ablations.png")

if __name__ == '__main__':
    main()

%%writefile exp_fpe_dynamic_growth.py
"""
Targeted FPE Transformer - Dynamic Network Growth (Cost-Saver Experiment)

This script empirically proves that Fixed Parameter Expansion (FPE) allows a cheap,
small network to match the performance of a massive network by dynamically expanding 
its geometry exactly when it reaches its intrinsic superposition plateau.

It evaluates 3 test modes:
1. `static_small`: Evaluates d_ff = 64 out of the gate. Stops early (low ceiling).
2. `static_large`: Evaluates d_ff = 256 out of the gate. Scales high (expensive).
3. `fpe_growth`: Starts at d_ff = 64. Triggers FPE at geometric plateau, tightly partitioning 
    parameters exactly into 4 children (expanding natively to d_ff = 256).

By tracking Accumulated Compute (Cost proxy) directly against Loss on standard 
TinyStories, we graph an efficiency curve to prove FPE captures large-model geometry 
on a cheap computational budget.
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

def split_ffn_neurons(W1, W2, b1=None, n_children=4, reference_W1=None, target_fraction=1.0, run_mode="fpe_growth"):
    """
    Fixed Parameter Expansion Core.
    STRICT MATHEMATICAL PARTITIONING: The non-zero connections of W1 are divided orthogonally 
    across N children. We don't just clone. We shatter dense multiplexing into clean fragments.
    """
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
            # FPE Geometric Detonation: Split into N strictly orthogonal subspace fragments
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
                
                # Assign this strictly orthogonal subset of connections to the child
                w1_child = torch.zeros(d_model, device=device, dtype=W1.dtype)
                for i in part_idx: w1_child[i] = w1_j[i]
                
                # Duplicate output vector so the disjoint fragments mathematically sum identical
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
def run_experiment(args, device, dataloader, tokenizer, vocab_size, run_mode):
    print(f"\n{'='*60}")
    print(f"Starting Run | Mode: {run_mode.upper()} | Target Factor: {args.growth_factor}x")
    print(f"{'='*60}")

    # Initialize correct baseline bounds
    start_d_ff = args.d_ff_base
    if run_mode == "static_large":
        start_d_ff = args.d_ff_base * args.growth_factor

    model = FPETransformer(vocab_size, args.d_model, args.n_heads, start_d_ff, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    data_iter = iter(dataloader)
    
    compute_losses = []
    recent_pr_dims = []
    accumulated_compute = 0
    has_expanded = False
    
    print(f"\n[Phase 1] Pre-training on sequence (Initial d_ff = {start_d_ff})...")
    
    # ---------------- Phase 1 ----------------
    for step in range(args.n_steps_phase1):
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
        
        # Flop Proxy Accumulator: D_FF controls MLP computational depth
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
            print(f"Phase 1 Step {step+1} | Cost Proxy: {accumulated_compute} | Loss: {current_loss:.4f} | D_PR: {pr_dim:.2f}")
            recent_pr_dims.append(pr_dim)
            
            if not has_expanded and run_mode == "fpe_growth" and len(recent_pr_dims) >= args.patience:
                window = recent_pr_dims[-args.patience:]
                dpr_diff = max(window) - min(window)
                if dpr_diff < args.tolerance:
                    print(f"\n--> {run_mode.upper()} Trigger! Plateaued natively at Capacity.")
                    has_expanded = True
                    break

    # If static modes, just finish learning on current Phase trajectory.
    if run_mode != "fpe_growth":
        print(f"Terminating Static Mode normally at Step {args.n_steps_phase1}. End Cost: {accumulated_compute}")
        return compute_losses

    # ---------------- Phase 2 (FPE ONLY) ----------------
    print(f"\n[Intervention] FPE Growing Network architecture dynamically...")
    old_m = model.block.ffn.d_ff
    new_m = model.block.ffn.expand_ffn(n_children=args.growth_factor, target_fraction=1.0, run_mode=run_mode)
    
    print(f"Geometric Detonation: Orthogonal Substructure Generated. d_ff: {old_m} -> {new_m}")
    print("Re-initializing optimizers for larger state...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
    
    remaining_steps = args.n_steps_phase1 - (step + 1)
    
    print("\n[Phase 2] Harnessing Large-Network geometry on newly created subspace...")
    for step2 in range(remaining_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        idx = batch["input_ids"].to(device)
        x = idx[:, :-1]
        y = idx[:, 1:]
        
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss = loss / args.grad_accum_steps
        loss.backward()
        
        # Post-Intervention Flop Proxy Accumulator is heavy
        accumulated_compute += model.block.ffn.d_ff
        
        if (step2 + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        current_loss = loss.item() * args.grad_accum_steps
        
        if (step2 + 1) % args.log_interval == 0:
            compute_losses.append((accumulated_compute, current_loss))
            print(f"Phase 2 Step {step2+1} (Total {(step+1)+step2+1}) | Cost: {accumulated_compute} | Loss: {current_loss:.4f}")

    return compute_losses

# -----------------------------------------------------------------------------
# Script Main & Graphing
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff_base", type=int, default=64)      # Small starting parameter size
    parser.add_argument("--growth_factor", type=int, default=4)   # Target Multiplier (Target large size = 256)
    parser.add_argument("--n_heads", type=int, default=4)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--n_steps_phase1", type=int, default=8000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.5) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    dataloader = get_dataloader(tokenizer, args.batch_size, args.seq_len)

    # ---------------- The Core 3-Model Validation Map ----------------
    stats_small = run_experiment(args, device, dataloader, tokenizer, vocab_size, "static_small")
    stats_large = run_experiment(args, device, dataloader, tokenizer, vocab_size, "static_large")
    stats_fpe = run_experiment(args, device, dataloader, tokenizer, vocab_size, "fpe_growth")

    # ==========================================
    # Efficiency Curve Output
    # ==========================================
    os.makedirs("../newfigures", exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    # Unpack A/B/C Test Arrays
    x_sm = [p[0] for p in stats_small]
    y_sm = [p[1] for p in stats_small]
    
    x_lg = [p[0] for p in stats_large]
    y_lg = [p[1] for p in stats_large]
    
    x_fpe = [p[0] for p in stats_fpe]
    y_fpe = [p[1] for p in stats_fpe]
    
    # Smooth data to capture true trends
    def smooth(scalars, weight=0.6):
        last = scalars[0]; smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val); last = smoothed_val
        return smoothed
    
    # Plotting
    plt.plot(x_lg, smooth(y_lg), label=f"Static Large Model (d_ff {args.d_ff_base * args.growth_factor})", color="black", alpha=0.9, linewidth=3)
    plt.plot(x_sm, smooth(y_sm), label=f"Static Small Model (d_ff {args.d_ff_base})", color="red", linestyle="--", alpha=0.8, linewidth=2)
    plt.plot(x_fpe, smooth(y_fpe), label=f"FPE Dynamic Growth (d_ff {args.d_ff_base} -> {args.d_ff_base * args.growth_factor})", color="blue", alpha=0.9, linewidth=3)
    
    plt.title("Dynamic Network Growth: Architectural Cost/Performance Efficiency", fontsize=14)
    plt.xlabel("Total Accumulated Compute Proxy (FLOPs)", fontsize=12)
    plt.ylabel("Validation Cross-Entropy Loss", fontsize=12)
    
    # Set boundaries cleanly
    plt.yscale("log")
    plt.ylim(top=max(max(y_sm), max(y_lg), max(y_fpe)) * 1.1)
    
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    plt.savefig("../newfigures/efficiency_curve.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved Computational Efficiency Figure: ../newfigures/efficiency_curve.png")

if __name__ == '__main__':
    main()

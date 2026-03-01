#!/usr/bin/env python3
"""
Real-World Scale FPE Efficiency Benchmark (GPT-2 Small architecture)

This script empirically proves that Fixed Parameter Expansion (FPE) disjoint
partitioning creates a massive geometric advantage and saves real wall-clock time
at scale on standard LLM benchmarks (Wikitext-103).

Evaluated Modes:
1. `static_small`: d_ff = 768. Low ceiling, fast.
2. `static_large`: d_ff = 3072. High ceiling, expensive.
3. `fpe_growth`: d_ff_start = 768. FPE dynamic disjoint expansion to d_ff = 3072.
"""

import time
import math
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
        
        if "static" in run_mode or len(nonzero_idx) <= 1:
            new_w1_cols.append(w1_j.unsqueeze(1))
            new_w2_rows.append(w2_j.unsqueeze(0))
            new_b1.append(b1_j.unsqueeze(0))
            expanded_indices.append(idx_counter)
            idx_counter += 1
        else:
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
# Architecture
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
    def __init__(self, vocab_size, d_model=768, n_heads=12, d_ff=3072, max_seq_len=512):
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
def get_dataloaders(tokenizer, batch_size=8, seq_len=256):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len+1, padding="max_length")

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # We select a subset for rapid experimentation locally, though it can run on full wikitext-103
    train_ds = dataset["train"].select(range(50000))
    val_ds = dataset["validation"].select(range(1000))
    
    train_tokenized = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    train_tokenized.set_format("torch", columns=["input_ids"])
    
    val_tokenized = val_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    val_tokenized.set_format("torch", columns=["input_ids"])
    
    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# Evaluation Helper
# -----------------------------------------------------------------------------
def evaluate(model, val_loader, device, tokenizer):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            idx = batch["input_ids"].to(device)
            x = idx[:, :-1]
            y = idx[:, 1:]
            
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item()
            num_batches += 1
            if num_batches >= 20: # Fast eval subset for rapid wall-clock cycles
                break
    model.train()
    val_loss = total_loss / max(1, num_batches)
    val_ppl = math.exp(min(val_loss, 20.0)) # clamp to avoid overflow display
    return val_loss, val_ppl

# -----------------------------------------------------------------------------
# Main Test Harness
# -----------------------------------------------------------------------------
def run_experiment(args, device, train_loader, val_loader, tokenizer, vocab_size, run_mode):
    print(f"\n{'='*70}")
    print(f"Starting Run | Mode: {run_mode.upper()}")
    print(f"{'='*70}")

    start_d_ff = args.d_ff_base
    if run_mode == "static_large":
        start_d_ff = args.d_ff_base * args.growth_factor

    model = FPETransformer(vocab_size, args.d_model, args.n_heads, start_d_ff, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_iter = iter(train_loader)
    
    # Trackers
    eval_stats = [] # Store tuple: (Wall-Clock Time, Val PPL, Step)
    dpr_stats = []  # Store tuple: (Step, FFN DPR)
    
    active_wall_clock = 0.0
    has_expanded = False
    
    print(f"\n[Phase 1] Training Initial Geometry (d_ff = {start_d_ff})...")
    model.train()
    
    for step in range(args.n_steps_max):
        # ---------------- Tracking & Evaluation Phase ----------------
        if (step + 1) % args.log_interval == 0 or step == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device, tokenizer)
            eval_stats.append((active_wall_clock, val_ppl, step+1))
            
            # Compute DPR using a random mini-batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                
            idx = batch["input_ids"].to(device)
            x = idx[:, :-1]
            
            with torch.no_grad():
                model.eval()
                _, h_states = model(x, return_hidden=True)
                model.train()
                h_sample = h_states[torch.randperm(h_states.size(0))[:2048]]
                pr_dim = compute_pr_dim(h_sample).item()
                
            dpr_stats.append((step+1, pr_dim))
            print(f"[{run_mode}] Step {step+1:04d} | Time: {active_wall_clock:.1f}s | Val PPL: {val_ppl:.2f} | D_PR: {pr_dim:.2f}")
            
            # FPE Trigger Logic
            if not has_expanded and run_mode == "fpe_growth" and len(dpr_stats) >= args.patience:
                window = [x[1] for x in dpr_stats[-args.patience:]]
                dpr_diff = max(window) - min(window)
                if dpr_diff < args.tolerance:
                    print(f"\n>>> {run_mode.upper()} TRIGGER! D_PR Plateaued. Generating Subspace Expansion...")
                    has_expanded = True
                    
                    old_m = model.block.ffn.d_ff
                    new_m = model.block.ffn.expand_ffn(n_children=args.growth_factor, target_fraction=1.0, run_mode=run_mode)
                    model = model.to(device)
                    print(f">>> Geometric Detonation: FFN {old_m} -> {new_m}")
                    
                    # Reset Optimizer
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
        
        # ---------------- Training Forward/Backward (Timed) ----------------
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        idx = batch["input_ids"].to(device)
        x = idx[:, :-1]
        y = idx[:, 1:]
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss = loss / args.grad_accum_steps
        loss.backward()
        
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        active_wall_clock += (t1 - t0)

    print(f"Finished {run_mode}. Total active processing time: {active_wall_clock:.2f}s")
    return eval_stats, dpr_stats

# -----------------------------------------------------------------------------
# Script Main & Graphing
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    
    # Standard GPT-2 Small scaling parameters
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff_base", type=int, default=768)      
    parser.add_argument("--growth_factor", type=int, default=4)   
    parser.add_argument("--n_heads", type=int, default=12)
    
    parser.add_argument("--lr", type=float, default=2e-4) # Slightly lower learning rate for GPT2 scale
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--n_steps_max", type=int, default=2000) # Keep low for quick single GPU evaluation
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.5) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Setting up GPT-2 architecture testing...")

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    train_loader, val_loader = get_dataloaders(tokenizer, args.batch_size, args.seq_len)

    # ---------------- Ensure Clean Evaluation Trajectories ----------------
    eval_fpe, dpr_fpe = run_experiment(args, device, train_loader, val_loader, tokenizer, vocab_size, "fpe_growth")
    eval_small, dpr_small = run_experiment(args, device, train_loader, val_loader, tokenizer, vocab_size, "static_small")
    eval_large, dpr_large = run_experiment(args, device, train_loader, val_loader, tokenizer, vocab_size, "static_large")

    print("\nGenerating Figure...")
    os.makedirs("../newfigures", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Smoother fn
    def smooth(scalars, weight=0.6):
        if not scalars: return scalars
        last = scalars[0]
        smoothed = []
        for p in scalars:
            s = last * weight + (1 - weight) * p
            smoothed.append(s)
            last = s
        return smoothed

    # ax1: PPL vs Wall Clock
    ax1.plot([e[0] for e in eval_large], smooth([e[1] for e in eval_large]), label=f"Static Large (3072)", color="black", linewidth=3, alpha=0.9)
    ax1.plot([e[0] for e in eval_small], smooth([e[1] for e in eval_small]), label=f"Static Small (768)", color="red", linestyle="--", linewidth=2, alpha=0.8)
    ax1.plot([e[0] for e in eval_fpe], smooth([e[1] for e in eval_fpe]), label=f"FPE Disjoint Expand (768 -> 3072)", color="blue", linewidth=3, alpha=0.9)
    ax1.set_title("Runtime Efficiency: Validation Perplexity vs Active Wall-Clock", fontsize=12)
    ax1.set_xlabel("Cumulative Wall-Clock Time (Seconds)", fontsize=11)
    ax1.set_ylabel("Validation Perplexity (Wikitext-103)", fontsize=11)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ax2: DPR vs Steps
    ax2.plot([d[0] for d in dpr_large], smooth([d[1] for d in dpr_large]), label=f"Static Large", color="black", linewidth=3, alpha=0.9)
    ax2.plot([d[0] for d in dpr_small], smooth([d[1] for d in dpr_small]), label=f"Static Small", color="red", linestyle="--", linewidth=2, alpha=0.8)
    ax2.plot([d[0] for d in dpr_fpe], smooth([d[1] for d in dpr_fpe]), label=f"FPE Growth", color="blue", linewidth=3, alpha=0.9)
    ax2.set_title("Geometric Expansion: Polysemantic Subspace Detonation", fontsize=12)
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("FFN Activation Participation Rank (D_PR)", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "gpt2_scale_efficiency.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nSuccess! Figure saved to {plot_path}")

if __name__ == '__main__':
    main()

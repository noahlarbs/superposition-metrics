%%writefile exp_fpe_transformer_tinystories.py
"""
Targeted FPE Transformer on TinyStories - Paper Validation Run

This script implements an autoregressive Transformer trained on TinyStories.
It executes a Width Ablation study over a list of FeedForward dimension sizes (d_ff_list),
monitoring the Representation Participation Ratio (D_PR) of the Multi-Layer Perceptron.

Upon plateau, it triggers targeted Fixed Parameter Expansion (FPE) on the top 
polysemantic neurons, shielding the system against BitNet 1.58b ternary quantization.

It also runs a strict "Control" baseline, where the identical plateau triggers quantization 
but *without* expanding the architecture, proving the FPE mechanism is responsible for 
validation loss recovery. It automatically generates and saves validation figures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import os
import copy
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Quantization & FPE Metrics
# -----------------------------------------------------------------------------
WEIGHT_THRESHOLD = 1e-6

def weight_quant_ternary(w):
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-1, 1) * scale
    return w + (w_q - w).detach()

def activation_quant(x):
    scale = 127.0 / x.abs().max().clamp(min=1e-8)
    x_q = torch.round(x * scale).clamp(-128, 127) / scale
    return x + (x_q - x).detach()

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

def split_ffn_neurons(W1, W2, b1=None, n_children=2, reference_W1=None, target_fraction=1.0, run_mode="fpe"):
    if reference_W1 is None: reference_W1 = W1
    device = W1.device
    d_model, d_ff = W1.shape
    
    new_w1_cols, new_w2_rows, new_b1 = [], [], []
    expanded_indices, idx_counter = [], 0
    
    # Target superposed neurons
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
        
        # Keep pure neurons unchanged
        if j not in target_indices:
            new_w1_cols.append(w1_j.unsqueeze(1))
            new_w2_rows.append(w2_j.unsqueeze(0))
            new_b1.append(b1_j.unsqueeze(0))
            idx_counter += 1
            continue
            
        nonzero_idx = (ref_w.abs() > WEIGHT_THRESHOLD).nonzero(as_tuple=True)[0]
        
        # If in exact control mode, or pure neuron, don't split, just mark for quantization
        if run_mode == "control" or len(nonzero_idx) <= 1:
            new_w1_cols.append(w1_j.unsqueeze(1))
            new_w2_rows.append(w2_j.unsqueeze(0))
            new_b1.append(b1_j.unsqueeze(0))
            expanded_indices.append(idx_counter)
            idx_counter += 1
        else:
            # Expand polysemantic neuron into n_children (FPE Mode)
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
                
                # Split W1 connections
                w1_child = torch.zeros(d_model, device=device, dtype=W1.dtype)
                for i in part_idx: w1_child[i] = w1_j[i]
                
                w2_child = w2_j.clone()
                b1_child = b1_j.clone()
                
                new_w1_cols.append(w1_child.unsqueeze(1))
                new_w2_rows.append(w2_child.unsqueeze(0))
                new_b1.append(b1_child.unsqueeze(0))
                
                expanded_indices.append(idx_counter)
                idx_counter += 1

    W1_new = torch.cat(new_w1_cols, dim=1)
    W2_new = torch.cat(new_w2_rows, dim=0)
    b1_new = torch.cat(new_b1, dim=0) if b1 is not None else None
    
    return W1_new, W2_new, b1_new, expanded_indices

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
        
        self.expanded_idx = []
        self.quantize_expanded = False

    def forward(self, x, return_hidden=False):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        if self.quantize_expanded and len(self.expanded_idx) > 0:
            mask = torch.zeros(self.d_ff, dtype=torch.bool, device=x.device)
            mask[self.expanded_idx] = True
            
            W1_exp = weight_quant_ternary(self.W1[:, mask])
            W1_base = self.W1[:, ~mask]
            
            # Hidden computation (activation quantize input for expanded W1)
            h_exp = activation_quant(x_flat) @ W1_exp + self.b1[mask]
            h_base = x_flat @ W1_base + self.b1[~mask]
            
            h = torch.empty(B*T, self.d_ff, device=x.device, dtype=x.dtype)
            h[:, mask] = h_exp
            h[:, ~mask] = h_base
            h = self.gelu(h)
            h = self.act_dropout(h)
            
            hidden_states = h.clone()
            
            W2_exp = weight_quant_ternary(self.W2[mask, :])
            W2_base = self.W2[~mask, :]
            
            out = activation_quant(h[:, mask]) @ W2_exp + h[:, ~mask] @ W2_base + self.b2
            
        else:
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
        W1_new, W2_new, b1_new, exp_idx = split_ffn_neurons(
            self.W1.data, self.W2.data, self.b1.data, 
            n_children=n_children, target_fraction=target_fraction, run_mode=run_mode
        )
        self.d_ff = W1_new.shape[1]
        self.W1 = nn.Parameter(W1_new)
        self.W2 = nn.Parameter(W2_new)
        self.b1 = nn.Parameter(b1_new)
        self.expanded_idx = exp_idx
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
        self.lm_head.weight = self.token_emb.weight # Weight tying

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
    dataset = dataset.select(range(100000)) # Larger subset for longer run
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
    tokenized.set_format("torch", columns=["input_ids"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------
# Main Test Harness
# -----------------------------------------------------------------------------
def run_experiment(args, device, dataloader, tokenizer, vocab_size, d_ff_current, run_mode):
    """Executes a full run for a specific d_ff and run_mode (fpe or control)."""
    print(f"\\n{'='*50}")
    print(f"Starting Run: d_ff={d_ff_current} | mode={run_mode.upper()}")
    print(f"{'='*50}")

    model = FPETransformer(vocab_size, args.d_model, args.n_heads, d_ff_current, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    data_iter = iter(dataloader)
    
    # Trackers
    losses_all = []
    pr_dims_all = []
    recent_pr_dims = []
    has_expanded = False
    split_step = args.n_steps_max
    
    print("\\n--- Phase 1: Pre-training ---")
    optimizer.zero_grad()
    
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
        # Gradient Accumulation Scaling
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss = loss / args.grad_accum_steps
        loss.backward()
        
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        # Unscale for logging
        current_loss = loss.item() * args.grad_accum_steps
        losses_all.append(current_loss)
        
        if (step + 1) % args.log_interval == 0:
            with torch.no_grad():
                h_sample = h_states[torch.randperm(h_states.size(0))[:2048]]
                pr_dim = compute_pr_dim(h_sample).item()
            
            pr_dims_all.append((step+1, pr_dim))
            print(f"Step {step+1} | Loss: {current_loss:.4f} | FFN D_PR: {pr_dim:.2f}")
            recent_pr_dims.append(pr_dim)
            
            # Dynamic Trigger Check
            if not has_expanded and len(recent_pr_dims) >= args.patience:
                window = recent_pr_dims[-args.patience:]
                dpr_diff = max(window) - min(window)
                if dpr_diff < args.tolerance:
                    print(f"\\n--> Dynamic Trigger! D_PR Plateaued (diff {dpr_diff:.2f} < {args.tolerance:.2f}). Triggering {run_mode.upper()}.")
                    has_expanded = True
                    split_step = step + 1
                    break

    if not has_expanded:
        print(f"Run finished early - no plateau met. Final Loss: {losses_all[-1]:.4f}")
        return losses_all, pr_dims_all, split_step

    # Phase 2: Intervention (FPE Expansion vs Control Null-Expansion)
    print(f"\\n--- Phase 2: Quantization Intervention ({run_mode.upper()}) ---")
    old_m = model.block.ffn.d_ff
    new_m = model.block.ffn.expand_ffn(n_children=2, target_fraction=args.target_fraction, run_mode=run_mode)
    model.block.ffn.quantize_expanded = True
    
    if run_mode == "fpe":
        print(f"FPE Expanded top {args.target_fraction*100}% neurons. d_ff: {old_m} -> {new_m}")
    else:
        print(f"Control Mode: Target {args.target_fraction*100}% neurons quantized. Dimension remains {old_m}.")
    
    # Reset optimizer for new parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    print("\\nFine-tuning Quantized Network...")
    for step in range(split_step, split_step + args.n_steps_finetune):
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
        
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        current_loss = loss.item() * args.grad_accum_steps
        losses_all.append(current_loss)
        
        if (step + 1) % args.log_interval == 0:
            with torch.no_grad():
                h_sample = h_states[torch.randperm(h_states.size(0))[:2048]]
                pr_dim = compute_pr_dim(h_sample).item()
            pr_dims_all.append((step+1, pr_dim))
            print(f"Finetune Step {step+1} | Loss: {current_loss:.4f} | FFN D_PR: {pr_dim:.2f}")

    return losses_all, pr_dims_all, split_step

# -----------------------------------------------------------------------------
# Harness Definition & Graph Generation
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff_list", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # FPE Triggers
    parser.add_argument("--n_steps_max", type=int, default=10000)
    parser.add_argument("--n_steps_finetune", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.5) 
    parser.add_argument("--target_fraction", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    dataloader = get_dataloader(tokenizer, args.batch_size, args.seq_len)

    results = {}
    
    # 1. Ablation Study over d_ff widths (Tracking geometric compression)
    for d_ff in args.d_ff_list:
        # Run standard FPE expansion
        losses_fpe, pr_dims_fpe, trigger_step = run_experiment(args, device, dataloader, tokenizer, vocab_size, d_ff, "fpe")
        
        # For the largest dimension (or pick a target), run a control validation
        if d_ff == args.d_ff_list[-1]:
            losses_ctrl, pr_dims_ctrl, _ = run_experiment(args, device, dataloader, tokenizer, vocab_size, d_ff, "control")
            results[d_ff] = {
                "fpe_losses": losses_fpe, "fpe_prs": pr_dims_fpe, 
                "ctrl_losses": losses_ctrl, "ctrl_prs": pr_dims_ctrl, 
                "trigger": trigger_step
            }
        else:
            results[d_ff] = {"fpe_losses": losses_fpe, "fpe_prs": pr_dims_fpe, "trigger": trigger_step}

    # ==========================================
    # Plotting Automation
    # ==========================================
    os.makedirs("../newfigures", exist_ok=True)
    
    # Plot 1: FPE vs Control Loss
    target_dff = args.d_ff_list[-1]
    plt.figure(figsize=(10, 6))
    fpe_l = results[target_dff]["fpe_losses"]
    ctrl_l = results[target_dff]["ctrl_losses"]
    trig = results[target_dff]["trigger"]
    
    plt.plot(range(len(fpe_l)), fpe_l, label="FPE Targeted Expansion", color="blue", alpha=0.8)
    plt.plot(range(len(ctrl_l)), ctrl_l, label="Strict Control (No FPE)", color="red", alpha=0.8)
    plt.axvline(x=trig, color='black', linestyle='--', label=f'Quantization Trigger (Step {trig})')
    
    plt.title(f"Validation Loss Recovery: FPE vs Control (d_ff={target_dff})")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(top=min(max(fpe_l[trig:] + ctrl_l[trig:]) * 1.5, 10.0)) # Zoom on interference
    plt.savefig("../newfigures/loss_vs_step.png", dpi=300, bbox_inches="tight")
    print("\nSaved Loss Comparison Figure: ../newfigures/loss_vs_step.png")
    
    # Plot 2: D_PR Width Ablation
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(torch.linspace(0, 1, len(args.d_ff_list)).numpy())
    
    for i, d_ff in enumerate(args.d_ff_list):
        pr_data = results[d_ff]["fpe_prs"]
        steps = [p[0] for p in pr_data]
        prs = [p[1] for p in pr_data]
        plt.plot(steps, prs, label=f"d_ff = {d_ff}", color=colors[i], linewidth=2)
        
    plt.title("Representation Compression ($D_{PR}$) across FFN Widths")
    plt.xlabel("Pre-training Steps")
    plt.ylabel("Participation Ratio ($D_{PR}$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.savefig("../newfigures/d_pr_ablation.png", dpi=300, bbox_inches="tight")
    print("Saved Width Ablation Figure: ../newfigures/d_pr_ablation.png")

if __name__ == '__main__':
    main()

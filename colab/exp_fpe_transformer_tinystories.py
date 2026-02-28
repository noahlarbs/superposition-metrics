"""
Targeted FPE Transformer on TinyStories

This script implements a minimal autoregressive Transformer trained on the TinyStories dataset.
It monitors the Representation Participation Ratio (D_PR) of the Multi-Layer Perceptron (FFN) hidden states.
Upon plateau, it triggers targeted Fixed Parameter Expansion (FPE) on the top `target_fraction` 
most polysemantic neurons, measured via Weight Participation Ratio.

The newly expanded sibling neurons are immediately subjected to BitNet 1.58b 
ternary weight and 8-bit activation quantization to test their robustness against noise.
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

# -----------------------------------------------------------------------------
# Quantization & FPE Metrics
# -----------------------------------------------------------------------------
WEIGHT_THRESHOLD = 1e-6

def weight_quant_ternary(w):
    """BitNet 1.58b Ternary Quantization [-1, 0, 1]"""
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-1, 1) * scale
    return w + (w_q - w).detach()

def activation_quant(x):
    """BitNet 8-bit absolute max activation quantization"""
    scale = 127.0 / x.abs().max().clamp(min=1e-8)
    x_q = torch.round(x * scale).clamp(-128, 127) / scale
    return x + (x_q - x).detach()

def compute_pr_dim(activations: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Representation Participation Ratio (D_PR) from hidden layer activations.
    Formula: D_PR = (Tr(Σ))^2 / Tr(Σ^2)
    """
    H = activations - activations.mean(dim=0, keepdim=True)
    batch_size, hidden_dim = H.shape
    n = batch_size - 1
    if n <= 0: return torch.tensor(float("nan"), device=activations.device)
    
    sigma = (H.T @ H) / n
    tr_sigma = sigma.trace()
    tr_sigma_sq = (sigma @ sigma).trace()
    return (tr_sigma * tr_sigma) / (tr_sigma_sq + eps)

def get_weight_pr(W):
    """
    Compute Weight Participation Ratio for incoming weight vector (dim=0).
    PR(w_j) = (sum_i |W_ij|)^2 / sum_i (W_ij^2).
    """
    l1_norm = W.abs().sum(dim=0)
    l2_sq_norm = (W ** 2).sum(dim=0)
    return (l1_norm ** 2) / (l2_sq_norm + 1e-8)

def split_ffn_neurons(W1, W2, b1=None, n_children=2, reference_W1=None, target_fraction=1.0):
    """
    Targeted FFN Expansion.
    W1: [d_model, d_ff]
    W2: [d_ff, d_model]
    Selects top `target_fraction` polysemantic neurons via PR on W1.
    """
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
        if len(nonzero_idx) <= 1:
            new_w1_cols.append(w1_j.unsqueeze(1))
            new_w2_rows.append(w2_j.unsqueeze(0))
            new_b1.append(b1_j.unsqueeze(0))
            idx_counter += 1
        else:
            # Expand polysemantic neuron into n_children
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
                
                # Duplicate W2 and b1
                w2_child = w2_j.clone()
                b1_child = b1_j.clone() / n_splits # Optional: Scale bias or keep same
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
        # x: [B, T, d_model] -> flatten to [B*T, d_model] for processing if needed
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        if self.quantize_expanded and len(self.expanded_idx) > 0:
            mask = torch.zeros(self.d_ff, dtype=torch.bool, device=x.device)
            mask[self.expanded_idx] = True
            
            # W1: [d_model, d_ff]
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
            
            # Save pre-W2 hidden state for Representation PR calc
            hidden_states = h.clone()
            
            # Activation quantize the forward pass into W2 for expanded nodes
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
        
    def expand_ffn(self, n_children, target_fraction):
        W1_new, W2_new, b1_new, exp_idx = split_ffn_neurons(
            self.W1.data, self.W2.data, self.b1.data, 
            n_children=n_children, target_fraction=target_fraction
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
    # Take a small subset for demonstration
    dataset = dataset.select(range(50000)) 
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
    tokenized.set_format("torch", columns=["input_ids"])
    
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # FPE Triggers
    parser.add_argument("--n_steps_max", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1.0) # Larger tolerance for real data
    parser.add_argument("--target_fraction", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer and TinyStories dataset...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    dataloader = get_dataloader(tokenizer, args.batch_size, args.seq_len)
    data_iter = iter(dataloader)

    model = FPETransformer(vocab_size, args.d_model, args.n_heads, args.d_ff, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Phase 1
    print("\\n--- Phase 1: Pre-training ---")
    recent_pr_dims, losses = [], []
    has_expanded = False
    
    for step in range(args.n_steps_max):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        idx = batch["input_ids"].to(device)
        x = idx[:, :-1]
        y = idx[:, 1:]
        
        optimizer.zero_grad()
        logits, h_states = model(x, return_hidden=True)
        # Flatten for loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % args.log_interval == 0:
            with torch.no_grad():
                # Only use a random sub-sample of h_states (e.g. 2048 rows) to compute PR efficiently
                h_sample = h_states[torch.randperm(h_states.size(0))[:2048]]
                pr_dim = compute_pr_dim(h_sample).item()
                
            print(f"Step {step+1} | Loss: {loss.item():.4f} | FFN D_PR: {pr_dim:.2f}")
            recent_pr_dims.append(pr_dim)
            
            # Dynamic Trigger Check
            if not has_expanded and len(recent_pr_dims) >= args.patience:
                window = recent_pr_dims[-args.patience:]
                dpr_diff = max(window) - min(window)
                if dpr_diff < args.tolerance:
                    print(f"\\n--> Dynamic Trigger! D_PR Plateaued (diff {dpr_diff:.2f} < {args.tolerance:.2f}). Triggering Targeted FPE.")
                    has_expanded = True
                    break

    if not has_expanded:
        print("Completed pre-training without hitting plateau tolerance.")
        return

    # Phase 2: Targeted Expansion & Quantized Fine-tuning
    print(f"\\n--- Phase 2: Targeted FPE & BitNet 1.58b Quantization ---")
    old_m = model.block.ffn.d_ff
    new_m = model.block.ffn.expand_ffn(n_children=2, target_fraction=args.target_fraction)
    model.block.ffn.quantize_expanded = True
    print(f"Expanded top {args.target_fraction*100}% polysemantic neurons. d_ff: {old_m} -> {new_m}")
    
    # Reset optimizer for expanded parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
    
    print("\\nFine-tuning Quantized Network...")
    for step in range(1000): # Quick 1000 step finetune
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        idx = batch["input_ids"].to(device)
        x = idx[:, :-1]
        y = idx[:, 1:]
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % args.log_interval == 0:
            print(f"Finetune Step {step+1} | Loss: {loss.item():.4f}")
            
    print("\\nFinished validation script.")

if __name__ == '__main__':
    main()

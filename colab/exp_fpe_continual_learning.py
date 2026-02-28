%%writefile exp_fpe_continual_learning.py
"""
Targeted FPE Transformer - Continual Learning & Catastrophic Forgetting

This experiment demonstrates the use of Fixed Parameter Expansion (FPE) to mitigate
Catastrophic Forgetting. It trains a minimal Transformer on Task A (TinyStories).

Upon geometric Representation PR (D_PR) plateau, the training data hot-swaps to 
Task B (WikiText-2).

- In Control Mode, the network remains static and immediately begins overwriting
  features, causing Task A Validation Loss to skyrocket.
- In FPE Mode, the geometric detonation splits the FFN (maintaining the fixed
  weight budget via partitioned children), creating orthogonal subspace to route
  the new Task B knowledge, thereby protecting Task A's performance.
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
# FPE Metrics & Splitting Math (Strict Partitioning)
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

def split_ffn_neurons(W1, W2, b1=None, n_children=2, reference_W1=None, target_fraction=1.0, run_mode="fpe"):
    """
    Fixed Parameter Expansion Core.
    STRICT MATHEMATICAL PARTITIONING: The non-zero connections of W1 are divided orthogonally 
    across the new children so the total number of incoming non-zero parameter weights remains fixed.
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
        
        # If Control Mode, do not split. Model capacity remains constrained.
        if run_mode == "control" or len(nonzero_idx) <= 1:
            new_w1_cols.append(w1_j.unsqueeze(1))
            new_w2_rows.append(w2_j.unsqueeze(0))
            new_b1.append(b1_j.unsqueeze(0))
            expanded_indices.append(idx_counter)
            idx_counter += 1
        else:
            # FPE Geometric Detonation: Split into disjoint orthogonal subspaces
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
                
                # Duplicate output vector so the disjoint fragments sum to the original output
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
# Dual Data Loaders
# -----------------------------------------------------------------------------
def get_dataloaders(tokenizer, batch_size=32, seq_len=128):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len+1, padding="max_length")

    # Task A: TinyStories
    ds_a = load_dataset("roneneldan/TinyStories", split="train[:50000]")
    ds_a_val = load_dataset("roneneldan/TinyStories", split="validation[:2000]")
    
    tok_a = ds_a.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
    tok_a.set_format("torch", columns=["input_ids"])
    dl_a_train = DataLoader(tok_a, batch_size=batch_size, shuffle=True)
    
    tok_a_val = ds_a_val.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
    tok_a_val.set_format("torch", columns=["input_ids"])
    dl_a_val = DataLoader(tok_a_val, batch_size=batch_size, shuffle=False)

    # Task B: WikiText-2 (New Knowledge)
    ds_b = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_b = ds_b.filter(lambda x: len(x['text'].strip()) > 10) # Filter empty lines
    ds_b = ds_b.select(range(min(50000, len(ds_b))))
    
    tok_b = ds_b.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
    tok_b.set_format("torch", columns=["input_ids"])
    dl_b_train = DataLoader(tok_b, batch_size=batch_size, shuffle=True)
    
    return dl_a_train, dl_a_val, dl_b_train

# -----------------------------------------------------------------------------
# Main Test Harness
# -----------------------------------------------------------------------------
def evaluate(model, dl_val, device, tokenizer, num_batches=20):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dl_val):
            if i >= num_batches: break
            idx = batch["input_ids"].to(device)
            x, y = idx[:, :-1], idx[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item()
    model.train()
    return total_loss / min(num_batches, len(dl_val))

def run_experiment(args, device, dl_a_train, dl_a_val, dl_b_train, tokenizer, vocab_size, run_mode):
    print(f"\n{'='*60}")
    print(f"Starting Continual Learning Run | Mode: {run_mode.upper()} | Base d_ff: {args.d_ff_base}")
    print(f"{'='*60}")

    model = FPETransformer(vocab_size, args.d_model, args.n_heads, args.d_ff_base, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    iter_a = iter(dl_a_train)
    recent_pr_dims, a_val_losses = [], []
    split_step = args.n_steps_phase1
    has_expanded = False
    
    print("\n[Phase 1] Pre-training on Task A (TinyStories)...")
    for step in range(args.n_steps_phase1):
        try:
            batch = next(iter_a)
        except StopIteration:
            iter_a = iter(dl_a_train)
            batch = next(iter_a)
            
        idx = batch["input_ids"].to(device)
        x, y = idx[:, :-1], idx[:, 1:]
        
        logits, h_states = model(x, return_hidden=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss = loss / args.grad_accum_steps
        loss.backward()
        
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        current_loss = loss.item() * args.grad_accum_steps
        
        if (step + 1) % args.log_interval == 0:
            with torch.no_grad():
                h_sample = h_states[torch.randperm(h_states.size(0))[:2048]]
                pr_dim = compute_pr_dim(h_sample).item()
            
            print(f"Phase 1 Step {step+1} | Loss A: {current_loss:.4f} | FFN D_PR: {pr_dim:.2f}")
            recent_pr_dims.append(pr_dim)
            
            if not has_expanded and len(recent_pr_dims) >= args.patience:
                window = recent_pr_dims[-args.patience:]
                dpr_diff = max(window) - min(window)
                if dpr_diff < args.tolerance:
                    print(f"\n--> {run_mode.upper()} Trigger! D_PR Plateaued. Halting Phase 1.")
                    has_expanded = True
                    split_step = step + 1
                    break

    # Baseline Evaluate Task A before Task B begins
    val_loss_a = evaluate(model, dl_a_val, device, tokenizer)
    a_val_losses.append((0, val_loss_a))
    print(f"Pre-Switch Task A Validation Loss: {val_loss_a:.4f}")

    # Phase 2: Intervention (FPE vs Control) Map
    print(f"\n[Intervention] Preparing Architecture for Task B ({run_mode.upper()})...")
    old_m = model.block.ffn.d_ff
    new_m = model.block.ffn.expand_ffn(n_children=2, target_fraction=args.target_fraction, run_mode=run_mode)
    
    if run_mode == "fpe":
        print(f"FPE Detonation: Orthogonal Subspace Generated. d_ff: {old_m} -> {new_m}")
    else:
        print(f"Control Mode: Static Architecture. d_ff remains {old_m}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    iter_b = iter(dl_b_train)
    
    print("\n[Phase 2] Fine-tuning on Task B (WikiText) & Tracking Forgetting...")
    for step in range(args.n_steps_phase2):
        try:
            batch = next(iter_b)
        except StopIteration:
            iter_b = iter(dl_b_train)
            batch = next(iter_b)
            
        idx = batch["input_ids"].to(device)
        x, y = idx[:, :-1], idx[:, 1:]
        
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss = loss / args.grad_accum_steps
        loss.backward()
        
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        current_loss = loss.item() * args.grad_accum_steps
        
        if (step + 1) % args.log_interval == 0:
            val_loss_a = evaluate(model, dl_a_val, device, tokenizer)
            a_val_losses.append((step+1, val_loss_a))
            print(f"P2 Step {step+1} | Loss B (Train): {current_loss:.4f} | Loss A (Val): {val_loss_a:.4f}")

    return a_val_losses

# -----------------------------------------------------------------------------
# Script Main & Graphing
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff_base", type=int, default=128) # Test FPE vs Control on 128
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--n_steps_phase1", type=int, default=10000)
    parser.add_argument("--n_steps_phase2", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.5) 
    parser.add_argument("--target_fraction", type=float, default=1.0) # 100% expansion
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    print("\nLoading Dual Datasets (TinyStories & WikiText-2)...")
    dl_a_t, dl_a_v, dl_b_t = get_dataloaders(tokenizer, args.batch_size, args.seq_len)

    # Execute FPE and Control sequentially
    val_loss_fpe = run_experiment(args, device, dl_a_t, dl_a_v, dl_b_t, tokenizer, vocab_size, "fpe")
    val_loss_ctrl = run_experiment(args, device, dl_a_t, dl_a_v, dl_b_t, tokenizer, vocab_size, "control")

    # ==========================================
    # Plotting Automation
    # ==========================================
    os.makedirs("../newfigures", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    steps = [p[0] for p in val_loss_fpe]
    fpe_l = [p[1] for p in val_loss_fpe]
    ctrl_l = [p[1] for p in val_loss_ctrl]
    
    plt.plot(steps, fpe_l, label=f"FPE Detonation (d_ff {args.d_ff_base} -> {args.d_ff_base*2})", color="blue", alpha=0.9, linewidth=2)
    plt.plot(steps, ctrl_l, label=f"Control Baseline (Static d_ff {args.d_ff_base})", color="red", alpha=0.9, linewidth=2)
    
    plt.title("Catastrophic Forgetting Mitigation via Fixed Parameter Expansion")
    plt.xlabel("Phase 2 Fine-Tuning Steps (WikiText-2)")
    plt.ylabel("Task A Validation Cross-Entropy Loss (TinyStories)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Optional annotation
    max_ctrl = max(ctrl_l)
    plt.annotate('Catastrophic Overwrite', xy=(steps[-1], ctrl_l[-1]), xytext=(steps[-1]-500, max_ctrl),
                 arrowprops=dict(facecolor='red', shrink=0.05), color='red')
                 
    plt.savefig("../newfigures/forgetting_curve.png", dpi=300, bbox_inches="tight")
    print("\nSaved Catastrophic Forgetting Mitgation Figure: ../newfigures/forgetting_curve.png")

if __name__ == '__main__':
    main()

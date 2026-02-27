"""
Advanced Quantized Fixed Parameter Expansion
Includes tracking for KL Divergence, BitNet LR Schedulers, and iq2_xxs / q2_k quantization.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
import argparse
import os
import copy
from adamw import AdamW
from pr_dim import compute_pr_dim

WEIGHT_THRESHOLD = 1e-6

def weight_quant_ternary(w):
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-1, 1) * scale
    return w + (w_q - w).detach()

def weight_quant_iq2_xxs(w):
    scale = w.abs().mean().clamp(min=1e-8) * 2.5
    norm_w = (w / scale).clamp(-1, 1)
    signs = torch.sign(norm_w)
    abs_w = norm_w.abs()
    w_q_abs = torch.where(abs_w > 0.66, torch.ones_like(abs_w), torch.ones_like(abs_w) / 3.0)
    w_q = signs * w_q_abs * scale
    return w + (w_q - w).detach()

def weight_quant_q2_k(w, block_size=32):
    original_shape = w.shape
    w_flat = w.flatten()
    n_elements = w_flat.numel()
    
    pad_size = (block_size - (n_elements % block_size)) % block_size
    if pad_size > 0:
        w_flat = F.pad(w_flat, (0, pad_size))
        
    blocks = w_flat.view(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    norm_blocks = blocks / scales
    
    _, top_indices = torch.topk(norm_blocks.abs(), min(2, block_size), dim=1)
    top_mask = torch.zeros_like(norm_blocks, dtype=torch.bool)
    top_mask.scatter_(1, top_indices, True)
    
    q4_blocks = torch.round(norm_blocks * 7.0) / 7.0
    
    signs = torch.sign(norm_blocks)
    abs_nb = norm_blocks.abs()
    w_q_abs = torch.where(abs_nb > 0.66, torch.ones_like(abs_nb), torch.ones_like(abs_nb) / 3.0)
    q2_blocks = signs * w_q_abs
    
    q_blocks = torch.where(top_mask, q4_blocks, q2_blocks)
    w_q_flat = q_blocks * scales
    
    if pad_size > 0:
        w_q_flat = w_q_flat[:-pad_size]
        
    w_q = w_q_flat.view(original_shape)
    return w + (w_q - w).detach()

def activation_quant(x):
    scale = 127.0 / x.abs().max().clamp(min=1e-8)
    x_q = torch.round(x * scale).clamp(-128, 127) / scale
    return x + (x_q - x).detach()


class FeatureRecoveryAdvanced(nn.Module):
    def __init__(self, n, m, W=None, b=None, expanded_indices=None, quant_method="ternary"):
        super().__init__()
        self.n = n
        self.m = m
        if W is not None:
            self.W = nn.Parameter(W.clone())
        else:
            self.W = nn.Parameter(torch.randn(n, m) / math.sqrt(m))
        if b is not None:
            self.b = nn.Parameter(b.clone())
        else:
            self.b = nn.Parameter(torch.randn(n))
        self.relu = nn.ReLU()
        self.expanded_indices = expanded_indices if expanded_indices is not None else []
        self.quant_method = quant_method
        
    def quantize_w(self, w):
        if self.quant_method == "ternary":
            return weight_quant_ternary(w)
        elif self.quant_method == "iq2_xxs":
            return weight_quant_iq2_xxs(w)
        elif self.quant_method == "q2_k":
            return weight_quant_q2_k(w)
        return w

    def forward(self, x, return_hidden=False):
        if len(self.expanded_indices) > 0:
            mask = torch.zeros(self.m, device=self.W.device, dtype=torch.bool)
            mask[self.expanded_indices] = True
            
            W_exp_q = self.quantize_w(self.W[:, mask])
            W_continuous = self.W[:, ~mask]
            
            hidden_exp = activation_quant(x) @ W_exp_q
            hidden_cont = x @ W_continuous
            
            hidden = torch.empty(x.shape[0], self.m, device=x.device, dtype=x.dtype)
            hidden[:, mask] = hidden_exp
            hidden[:, ~mask] = hidden_cont
            
            hidden_for_out = hidden.clone()
            hidden_for_out[:, mask] = activation_quant(hidden[:, mask])
            
            output = self.relu(hidden_for_out[:, mask] @ W_exp_q.T + hidden_for_out[:, ~mask] @ W_continuous.T + self.b)
        else:
            hidden = x @ self.W
            output = self.relu(hidden @ self.W.T + self.b)
            
        if return_hidden:
            return output, hidden
        return output

def split_polysemantic_neurons(W, n_children=2, reference_W=None):
    if reference_W is None:
        reference_W = W
    device = W.device
    n, m = W.shape

    new_columns = []
    split_indices = []
    expanded_indices = []
    idx_counter = 0

    for j in range(m):
        w = W[:, j]
        ref_w = reference_W[:, j]
        nonzero_idx = (ref_w.abs() > WEIGHT_THRESHOLD).nonzero(as_tuple=True)[0]

        if len(nonzero_idx) <= 1:
            new_columns.append(w.unsqueeze(1))
            idx_counter += 1
        else:
            split_indices.append(j)
            idx_list = nonzero_idx.tolist()
            n_conn = len(idx_list)
            n_splits = min(n_children, n_conn)
            base_size = n_conn // n_splits
            remainder = n_conn % n_splits

            offset = 0
            for k in range(n_splits):
                size = base_size + (1 if k < remainder else 0)
                if size == 0:
                    continue
                part_idx = idx_list[offset : offset + size]
                offset += size

                w_child = torch.zeros(n, device=device, dtype=W.dtype)
                for i in part_idx:
                    w_child[i] = w[i]
                new_columns.append(w_child.unsqueeze(1))
                if k > 0:
                    expanded_indices.append(idx_counter)
                else:
                    expanded_indices.append(idx_counter)
                idx_counter += 1

    if not new_columns:
        return W, [], []

    W_new = torch.cat(new_columns, dim=1)
    return W_new, split_indices, expanded_indices

def get_lr_cooldown(step, lr, total_steps, warmup_steps=2000, cooldown_ratio=0.5, min_lr=1e-5):
    cooldown_start = int(total_steps * cooldown_ratio)
    if step >= cooldown_start:
        return min_lr
    else:
        eff_total = cooldown_start
        step = step + 1
        if step < warmup_steps:
             return lr * step / warmup_steps
        else:
             progress = (step - warmup_steps) / (eff_total - warmup_steps)
             return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) + min_lr

def get_lr(step, lr, n_steps, warmup_steps=2000):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_steps_pre", type=int, default=4000)
    parser.add_argument("--n_steps_post", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--quant_method", type=str, default="ternary", choices=["ternary", "iq2_xxs", "q2_k"])
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Quant: {args.quant_method}")

    if args.output is None:
        args.output = f"../outputs/exp_fpe_advanced_{args.quant_method}.pt"

    prob = torch.tensor([1.0 / i ** (1 + args.alpha) for i in range(1, args.n + 1)])
    prob = prob / prob.sum()
    prob = prob.to(device)

    n_steps_total = args.n_steps_pre + args.n_steps_post
    split_step = args.n_steps_pre

    model = FeatureRecoveryAdvanced(args.n, args.m).to(device)
    optimizer = AdamW([{"params": model.W, "weight_decay": -1.0}, {"params": model.b, "weight_decay": 0.0}], lr=1e-2)
    criteria = nn.MSELoss()

    losses = []
    pr_dims = []

    for step in range(split_step):
        x = (torch.rand(args.batch_size, args.n, device=device) < prob) * torch.rand(args.batch_size, args.n, device=device) * 2
        lr = get_lr(step, 1e-2, split_step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        optimizer.zero_grad()
        y, hidden = model(x, return_hidden=True)
        loss = criteria(y, x)
        loss.backward()
        optimizer.step()
        with torch.no_grad(): pr_dim = compute_pr_dim(hidden.detach()).item()
        losses.append(loss.item())
        pr_dims.append(pr_dim)
        if (step+1) % args.log_interval == 0:
            print(f"step {step+1}/{split_step} | Phase 1 Loss: {loss.item():.4e} D_PR: {pr_dim:.2f}")

    W_old = model.W.detach()
    W_new, split_indices, expanded_indices = split_polysemantic_neurons(W_old)
    m_new = W_new.shape[1]

    model_full = FeatureRecoveryAdvanced(args.n, m_new, W=W_new, b=model.b.detach(), expanded_indices=[]).to(device)
    optimizer_full = AdamW([{"params": model_full.W, "weight_decay": -1.0}, {"params": model_full.b, "weight_decay": 0.0}], lr=1e-2)
    
    model_quant = FeatureRecoveryAdvanced(args.n, m_new, W=W_new, b=model.b.detach(), expanded_indices=expanded_indices, quant_method=args.quant_method).to(device)
    optimizer_quant = AdamW([{"params": model_quant.W, "weight_decay": -1.0}, {"params": model_quant.b, "weight_decay": 0.0}], lr=1e-2)

    for opt, mod in [(optimizer_full, model_full), (optimizer_quant, model_quant)]:
        opt.zero_grad()
        mod(torch.zeros(1, args.n, device=device)).sum().backward()
        opt.step()

    if len(optimizer.state) > 0:
        old_state_W = optimizer.state[model.W]
        if 'exp_avg' in old_state_W:
            base_exp_avg, _, _ = split_polysemantic_neurons(old_state_W['exp_avg'], reference_W=W_old)
            base_exp_avg_sq, _, _ = split_polysemantic_neurons(old_state_W['exp_avg_sq'], reference_W=W_old)
            for opt, mod in [(optimizer_full, model_full), (optimizer_quant, model_quant)]:
                opt.state[mod.W]['exp_avg'] = base_exp_avg.clone()
                opt.state[mod.W]['exp_avg_sq'] = base_exp_avg_sq.clone()
                opt.state[mod.W]['step'] = old_state_W['step']
                opt.state[mod.b]['exp_avg'] = optimizer.state[model.b]['exp_avg'].clone()
                opt.state[mod.b]['exp_avg_sq'] = optimizer.state[model.b]['exp_avg_sq'].clone()
                opt.state[mod.b]['step'] = optimizer.state[model.b]['step']

    sparsity_mask = (model_full.W.abs() > 0).float()

    losses_full = list(losses)
    pr_dims_full = list(pr_dims)
    losses_quant = list(losses)
    pr_dims_quant = list(pr_dims)
    kl_divs = []

    for step in range(split_step, n_steps_total):
        x = (torch.rand(args.batch_size, args.n, device=device) < prob) * torch.rand(args.batch_size, args.n, device=device) * 2

        lr_full = get_lr(step, 1e-2, n_steps_total)
        lr_quant = get_lr_cooldown(step - split_step, 1e-2, args.n_steps_post, warmup_steps=int(args.n_steps_post*0.2))

        for pg in optimizer_full.param_groups: pg["lr"] = lr_full
            
        for pg in optimizer_quant.param_groups:
            pg["lr"] = lr_quant
            if step - split_step >= int(args.n_steps_post * 0.5):
                pg["weight_decay"] = 0.0

        optimizer_full.zero_grad()
        y_f, hidden_f = model_full(x, return_hidden=True)
        loss_f = criteria(y_f, x)
        loss_f.backward()
        model_full.W.grad.data *= sparsity_mask
        optimizer_full.step()

        optimizer_quant.zero_grad()
        y_q, hidden_q = model_quant(x, return_hidden=True)
        loss_q = criteria(y_q, x)
        loss_q.backward()
        model_quant.W.grad.data *= sparsity_mask
        optimizer_quant.step()

        with torch.no_grad():
            pr_dim_f = compute_pr_dim(hidden_f.detach()).item()
            pr_dim_q = compute_pr_dim(hidden_q.detach()).item()
            
            log_pseudo_q = F.log_softmax(y_q, dim=-1)
            pseudo_f = F.softmax(y_f, dim=-1)
            kl = F.kl_div(log_pseudo_q, pseudo_f, reduction='batchmean').item()

        losses_full.append(loss_f.item())
        pr_dims_full.append(pr_dim_f)
        losses_quant.append(loss_q.item())
        pr_dims_quant.append(pr_dim_q)
        kl_divs.append(kl)

        if (step+1) % args.log_interval == 0:
            print(f"step {step+1}/{n_steps_total} | F-Loss: {loss_f.item():.4e} F-PR: {pr_dim_f:.2f} | Q-Loss: {loss_q.item():.4e} Q-PR: {pr_dim_q:.2f} | KL: {kl:.4e}")

    x_eval = (torch.rand(args.eval_batch_size, args.n, device=device) < prob) * torch.rand(args.eval_batch_size, args.n, device=device) * 2
    with torch.no_grad():
        target_loss = criteria(model_full(x_eval), x_eval).item()
        quant_loss = criteria(model_quant(x_eval), x_eval).item()
        
    print(f"\nFinal Eval Target Loss: {target_loss:.4e} | Quant Loss: {quant_loss:.4e}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "args": vars(args),
        "losses_full": losses_full,
        "pr_dims_full": pr_dims_full,
        "losses_quant": losses_quant,
        "pr_dims_quant": pr_dims_quant,
        "kl_divs": kl_divs,
        "m_before": args.m,
        "m_after": m_new,
    }, args.output)

if __name__ == "__main__":
    main()

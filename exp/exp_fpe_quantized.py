"""
Fixed Parameter Expansion (FPE) experiment based on Subramanian et al. (2025).

Extends the strong superposition baseline (weight_decay=-1) with an intervention at step 4000:
split polysemantic neurons and quantize the resulting expanded neurons using 
BitNet 1.58b absmean ternary quantization for weights and 8-bit max-scaling for activations.
Trains a full-weight FPE model as a baseline target, then trains the quantized model
until it matches the target validation loss.
"""

import torch
from torch import nn
import math
import argparse
import os
import copy
from adamw import AdamW
from pr_dim import compute_pr_dim


# Threshold for "non-zero" weight (for identifying connections to partition)
WEIGHT_THRESHOLD = 1e-6


def weight_quant(w):
    """BitNet 1.58b weight quantization (absmean ternary) with STE."""
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-1, 1) * scale
    return w + (w_q - w).detach()


def activation_quant(x):
    """8-bit max-scaling activation quantization with STE."""
    scale = 127.0 / x.abs().max().clamp(min=1e-8)
    x_q = torch.round(x * scale).clamp(-128, 127) / scale
    return x + (x_q - x).detach()


class FeatureRecoveryQuantized(nn.Module):
    """FeatureRecovery that exposes hidden layer activations for PR dimension computation,
    and selectively applies BitNet 1.58b quantization to expanded neurons.
    """

    def __init__(self, n, m, W=None, b=None, expanded_indices=None):
        super(FeatureRecoveryQuantized, self).__init__()
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

    def forward(self, x, return_hidden=False):
        if len(self.expanded_indices) > 0:
            mask = torch.zeros(self.m, device=self.W.device, dtype=torch.bool)
            mask[self.expanded_indices] = True
            
            W_exp_q = weight_quant(self.W[:, mask])
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
    """
    Split polysemantic neurons by partitioning each parent's non-zero weights
    disjointly across children. Maintains total non-zero parameter count.

    Returns:
        W_new: [n, m_new] expanded weight matrix
        split_indices: list of old indices that were split
        expanded_indices: list of new indices belonging to the expanded children
    """
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
            # Monosemantic: keep as single neuron
            new_columns.append(w.unsqueeze(1))
            idx_counter += 1
        else:
            # Polysemantic: partition connections disjointly across children
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
                expanded_indices.append(idx_counter)
                idx_counter += 1

    if not new_columns:
        return W, [], []

    W_new = torch.cat(new_columns, dim=1)
    return W_new, split_indices, expanded_indices


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
    parser = argparse.ArgumentParser(description="FPE: Quantized Extended Parameter Expansion experiment")
    parser.add_argument("--n", type=int, default=1000, help="number of features")
    parser.add_argument("--m", type=int, default=50, help="hidden dimension (pre-split)")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument("--n_steps_pre", type=int, default=4000, help="steps before FPE intervention")
    parser.add_argument("--n_steps_post", type=int, default=2000, help="steps after FPE intervention")
    parser.add_argument("--max_extra_steps", type=int, default=5000, help="max steps to run quantized model to match target loss")
    parser.add_argument("--log_interval", type=int, default=500, help="log every N steps")
    parser.add_argument("--eval_batch_size", type=int, default=8192, help="batch size for evaluating target loss")
    parser.add_argument("--alpha", type=float, default=0.0, help="feature distribution power-law exponent")
    parser.add_argument("--output", type=str, default=None, help="output path. If not provided, dynamically generated.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.output is None:
        args.output = f"../outputs/exp_fpe_quantized_n{args.n}_m{args.m}_pre{args.n_steps_pre}_post{args.n_steps_post}.pt"

    # Feature distribution
    prob = torch.tensor([1.0 / i ** (1 + args.alpha) for i in range(1, args.n + 1)])
    prob = prob / prob.sum()
    prob = prob.to(device)

    n_steps_total = args.n_steps_pre + args.n_steps_post
    split_step = args.n_steps_pre

    # Strong superposition pre-intervention base model
    model = FeatureRecoveryQuantized(args.n, args.m).to(device)
    parameter_groups = [
        {"params": model.W, "weight_decay": -1.0},
        {"params": model.b, "weight_decay": 0.0},
    ]
    optimizer = AdamW(parameter_groups, lr=1e-2)
    criteria = nn.MSELoss()

    losses = []
    pr_dims = []

    # Baseline tracking
    baseline_model = FeatureRecoveryQuantized(args.n, args.m).to(device)
    baseline_model.load_state_dict(model.state_dict())
    baseline_optimizer = AdamW([
        {"params": baseline_model.W, "weight_decay": -1.0},
        {"params": baseline_model.b, "weight_decay": 0.0},
    ], lr=1e-2)
    
    losses_baseline = []
    pr_dims_baseline = []

    print(f"\n--- Phase 1: Strong superposition (steps 0–{split_step}) ---")

    for step in range(split_step):
        x = (
            (torch.rand(args.batch_size, args.n, device=device) < prob)
            * torch.rand(args.batch_size, args.n, device=device)
            * 2
        )

        lr = get_lr(step, 1e-2, split_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        y, hidden = model(x, return_hidden=True)
        loss = criteria(y, x)
        loss.backward()
        optimizer.step()

        # Baseline step
        baseline_optimizer.zero_grad()
        y_b, hidden_b = baseline_model(x, return_hidden=True)
        loss_b = criteria(y_b, x)
        loss_b.backward()
        baseline_optimizer.step()

        with torch.no_grad():
            pr_dim = compute_pr_dim(hidden.detach()).item()
            pr_dim_b = compute_pr_dim(hidden_b.detach()).item()

        losses.append(loss.item())
        pr_dims.append(pr_dim)
        losses_baseline.append(loss_b.item())
        pr_dims_baseline.append(pr_dim_b)

        if (step + 1) % args.log_interval == 0:
            print(f"  step {step + 1}/{split_step} | loss: {loss.item():.4e} D_PR: {pr_dim:.2f} | Base loss: {loss_b.item():.4e} D_PR: {pr_dim_b:.2f}")

    print(f"\n  [Before FPE] step {split_step} | loss: {losses[-1]:.4e} | D_PR: {pr_dims[-1]:.2f}")

    # === FPE INTERVENTION: Split polysemantic neurons ===
    W_old = model.W.detach()
    m_old = W_old.shape[1]
    nnz_before = (W_old.abs() > WEIGHT_THRESHOLD).sum().item()

    W_new, split_indices, expanded_indices = split_polysemantic_neurons(W_old)
    n_split = len(split_indices)
    m_new = W_new.shape[1]
    nnz_after = (W_new.abs() > WEIGHT_THRESHOLD).sum().item()

    print(f"\n--- FPE intervention at step {split_step} ---")
    print(f"  Hidden Layer Neurons: {m_old} -> {m_new} ({n_split} neurons split)")
    print(f"  Total expanded (quantized) neurons: {len(expanded_indices)}")
    print(f"  Non-zeros: {nnz_before} -> {nnz_after} (sparsity preserved)")

    # 1. Build FULL WEIGHT expanded model
    model_full = FeatureRecoveryQuantized(args.n, m_new, W=W_new, b=model.b.detach(), expanded_indices=[]).to(device)
    optimizer_full = AdamW([
        {"params": model_full.W, "weight_decay": -1.0},
        {"params": model_full.b, "weight_decay": 0.0},
    ], lr=1e-2)

    # 2. Build QUANTIZED expanded model
    model_quant = FeatureRecoveryQuantized(args.n, m_new, W=W_new, b=model.b.detach(), expanded_indices=expanded_indices).to(device)
    optimizer_quant = AdamW([
        {"params": model_quant.W, "weight_decay": -1.0},
        {"params": model_quant.b, "weight_decay": 0.0},
    ], lr=1e-2)

    # Initialize optimizers to set up dummy states (so we can copy state dict safely)
    for opt, mod in [(optimizer_full, model_full), (optimizer_quant, model_quant)]:
        opt.zero_grad()
        mod(torch.zeros(1, args.n, device=device)).sum().backward()
        opt.step()

    # Safely map momenum state from W_old to W_new
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


    # Enforce sparsity 
    sparsity_mask = (model_full.W.abs() > 0).float()

    losses_full = list(losses)
    pr_dims_full = list(pr_dims)
    losses_quant = list(losses)
    pr_dims_quant = list(pr_dims)

    print(f"\n--- Phase 2: Post-FPE Synchronous Training (steps {split_step+1}–{n_steps_total}) ---")

    for step in range(split_step, n_steps_total):
        x = (
            (torch.rand(args.batch_size, args.n, device=device) < prob)
            * torch.rand(args.batch_size, args.n, device=device)
            * 2
        )

        lr = get_lr(step, 1e-2, n_steps_total)
        for pg in optimizer_full.param_groups:
            pg["lr"] = lr
        for pg in optimizer_quant.param_groups:
            pg["lr"] = lr
        for pg in baseline_optimizer.param_groups:
            pg["lr"] = lr

        # Baseline step
        baseline_optimizer.zero_grad()
        y_b, hidden_b = baseline_model(x, return_hidden=True)
        loss_b = criteria(y_b, x)
        loss_b.backward()
        baseline_optimizer.step()

        # Full-weight step
        optimizer_full.zero_grad()
        y_f, hidden_f = model_full(x, return_hidden=True)
        loss_f = criteria(y_f, x)
        loss_f.backward()
        model_full.W.grad.data *= sparsity_mask
        optimizer_full.step()

        # Quantized step
        optimizer_quant.zero_grad()
        y_q, hidden_q = model_quant(x, return_hidden=True)
        loss_q = criteria(y_q, x)
        loss_q.backward()
        model_quant.W.grad.data *= sparsity_mask
        optimizer_quant.step()

        with torch.no_grad():
            pr_dim_f = compute_pr_dim(hidden_f.detach()).item()
            pr_dim_q = compute_pr_dim(hidden_q.detach()).item()
            pr_dim_b = compute_pr_dim(hidden_b.detach()).item()

        losses_full.append(loss_f.item())
        pr_dims_full.append(pr_dim_f)
        losses_quant.append(loss_q.item())
        pr_dims_quant.append(pr_dim_q)
        losses_baseline.append(loss_b.item())
        pr_dims_baseline.append(pr_dim_b)

        if (step + 1) % args.log_interval == 0:
            print(f"  step {step + 1}/{n_steps_total} | Full loss: {loss_f.item():.4e} D_PR: {pr_dim_f:.2f} | Quant loss: {loss_q.item():.4e} D_PR: {pr_dim_q:.2f} | Base loss: {loss_b.item():.4e} D_PR: {pr_dim_b:.2f}")

    # Evaluate target validation loss
    print("\n--- Evaluating Target Loss for Quantized Catch-up ---")
    x_eval = (torch.rand(args.eval_batch_size, args.n, device=device) < prob) * torch.rand(args.eval_batch_size, args.n, device=device) * 2
    
    with torch.no_grad():
        target_loss = criteria(model_full(x_eval), x_eval).item()
        quant_loss = criteria(model_quant(x_eval), x_eval).item()

    print(f"  Target loss (Full-weight): {target_loss:.6e}")
    print(f"  Current loss (Quantized):  {quant_loss:.6e}")

    # Phase 3: Catch-up training for Quantized model
    extra_steps = 0
    final_catchup_loss = quant_loss
    
    if quant_loss > target_loss:
        print(f"\n--- Phase 3: Running quantized model until val loss <= {target_loss:.6e} ---")
        
        # Keep LR at minimum LR (end of cosine schedule)
        min_lr = get_lr(n_steps_total - 1, 1e-2, n_steps_total)
        for pg in optimizer_quant.param_groups:
            pg["lr"] = min_lr
            
        while final_catchup_loss > target_loss and extra_steps < args.max_extra_steps:
            x = (
                (torch.rand(args.batch_size, args.n, device=device) < prob)
                * torch.rand(args.batch_size, args.n, device=device)
                * 2
            )
            optimizer_quant.zero_grad()
            y_q, hidden_q = model_quant(x, return_hidden=True)
            loss_q = criteria(y_q, x)
            loss_q.backward()
            model_quant.W.grad.data *= sparsity_mask
            optimizer_quant.step()
            
            with torch.no_grad():
                pr_dim_q = compute_pr_dim(hidden_q.detach()).item()
                
            losses_quant.append(loss_q.item())
            pr_dims_quant.append(pr_dim_q)
            
            extra_steps += 1
            
            if extra_steps % (args.log_interval // 5) == 0:
                with torch.no_grad():
                    # Evaluate strictly to compare
                    curr_eval_loss = criteria(model_quant(x_eval), x_eval).item()
                print(f"  extra step {extra_steps} | Eval loss: {curr_eval_loss:.4e} D_PR: {pr_dim_q:.2f}")
                final_catchup_loss = curr_eval_loss
                
        if final_catchup_loss <= target_loss:
            print(f"\n  Success! Quantized model matched target loss after {extra_steps} extra steps.")
        else:
            print(f"\n  Stopped: Quantized model failed to match target loss within {args.max_extra_steps} extra steps.")
    else:
         print("\n  Quantized model already matched target loss! No extra steps needed.")

    
    print("\n--- Final Summary ---")
    print(f"  Baseline:  loss={losses_baseline[-1]:.4e}, D_PR={pr_dims_baseline[-1]:.2f}")
    print(f"  Full FPE:  loss={losses_full[-1]:.4e}, D_PR={pr_dims_full[-1]:.2f}")
    print(f"  Quant FPE (step {n_steps_total + extra_steps}): loss={final_catchup_loss:.4e} (eval_loss), D_PR={pr_dims_quant[-1]:.2f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "losses_full": losses_full,
            "pr_dims_full": pr_dims_full,
            "losses_quant": losses_quant,
            "pr_dims_quant": pr_dims_quant,
            "losses_baseline": losses_baseline,
            "pr_dims_baseline": pr_dims_baseline,
            "extra_steps": extra_steps,
            "target_loss": target_loss,
            "final_catchup_loss": final_catchup_loss,
            "m_before": m_old,
            "m_after": m_new,
        },
        args.output,
    )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

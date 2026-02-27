"""
Fixed Parameter Expansion (FPE) experiment based on Subramanian et al. (2025).

Extends the strong superposition baseline (weight_decay=-1) with an intervention at step 4000:
split polysemantic neurons by partitioning each parent's weights disjointly across children,
maintaining the sparsity budget. Resumes training for 2000 steps to measure if disentangling
lowers intrinsic dimensionality and reduces interference.
"""

import torch
from torch import nn
import math
import argparse
import os
from adamw import AdamW
from pr_dim import compute_pr_dim


# Threshold for "non-zero" weight (for identifying connections to partition)
WEIGHT_THRESHOLD = 1e-6


class FeatureRecoveryWithHidden(nn.Module):
    """FeatureRecovery that exposes hidden layer activations for PR dimension computation."""

    def __init__(self, n, m, W=None, b=None):
        super(FeatureRecoveryWithHidden, self).__init__()
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

    def forward(self, x, return_hidden=False):
        hidden = x @ self.W
        output = self.relu(hidden @ self.W.T + self.b)
        if return_hidden:
            return output, hidden
        return output


def split_polysemantic_neurons(W, n_children=2, reference_W=None):
    """
    Split polysemantic neurons by partitioning each parent's non-zero weights
    disjointly across children. Maintains total non-zero parameter count.

    Args:
        W: [n, m] weight matrix
        n_children: number of children per parent (default 2)
        reference_W: Optional [n, m] matrix, used to determine non-zero partitions

    Returns:
        W_new: [n, m_new] expanded weight matrix
        n_split: number of neurons that were split
    """
    if reference_W is None:
        reference_W = W
    device = W.device
    n, m = W.shape

    new_columns = []
    split_indices = []
    for j in range(m):
        w = W[:, j]
        ref_w = reference_W[:, j]
        nonzero_idx = (ref_w.abs() > WEIGHT_THRESHOLD).nonzero(as_tuple=True)[0]

        if len(nonzero_idx) <= 1:
            # Monosemantic: keep as single neuron
            new_columns.append(w.unsqueeze(1))
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

    if not new_columns:
        return W, []

    W_new = torch.cat(new_columns, dim=1)
    return W_new, split_indices


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
    parser = argparse.ArgumentParser(description="FPE: Fixed Parameter Expansion experiment")
    parser.add_argument("--n", type=int, default=1000, help="number of features")
    parser.add_argument("--m", type=int, default=50, help="hidden dimension (pre-split)")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument("--n_steps_pre", type=int, default=4000, help="steps before FPE intervention")
    parser.add_argument("--n_steps_post", type=int, default=2000, help="steps after FPE intervention")
    parser.add_argument("--log_interval", type=int, default=500, help="log every N steps")
    parser.add_argument("--alpha", type=float, default=0.0, help="feature distribution power-law exponent")
    parser.add_argument("--output", type=str, default=None, help="output path. If not provided, dynamically generated.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.output is None:
        args.output = f"../outputs/exp_fpe_n{args.n}_m{args.m}_pre{args.n_steps_pre}_post{args.n_steps_post}.pt"

    # Feature distribution
    prob = torch.tensor([1.0 / i ** (1 + args.alpha) for i in range(1, args.n + 1)])
    prob = prob / prob.sum()
    prob = prob.to(device)

    n_steps_total = args.n_steps_pre + args.n_steps_post
    split_step = args.n_steps_pre

    # Strong superposition baseline (weight_decay=-1)
    model = FeatureRecoveryWithHidden(args.n, args.m).to(device)
    parameter_groups = [
        {"params": model.W, "weight_decay": -1.0},
        {"params": model.b, "weight_decay": 0.0},
    ]
    optimizer = AdamW(parameter_groups, lr=1e-2)
    criteria = nn.MSELoss()

    losses = []
    pr_dims = []
    delta_log = {}  # before_split, after_split, end

    print("\n--- Phase 1: Strong superposition (steps 0–4000) ---")

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

        with torch.no_grad():
            pr_dim = compute_pr_dim(hidden.detach()).item()

        losses.append(loss.item())
        pr_dims.append(pr_dim)

        if (step + 1) % args.log_interval == 0:
            print(f"  step {step + 1}/{split_step} | loss: {loss.item():.4e} | D_PR: {pr_dim:.2f}")

    # === BEFORE SPLIT (immediately before intervention) ===
    delta_log["before_split"] = {
        "loss": losses[-1],
        "pr_dim": pr_dims[-1],
        "step": split_step - 1,
    }
    print(f"\n  [Before FPE] step {split_step} | loss: {losses[-1]:.4e} | D_PR: {pr_dims[-1]:.2f}")

    # === FPE INTERVENTION: Split polysemantic neurons ===
    W_old = model.W.detach()
    m_old = W_old.shape[1]
    nnz_before = (W_old.abs() > WEIGHT_THRESHOLD).sum().item()

    W_new, split_indices = split_polysemantic_neurons(W_old)
    n_split = len(split_indices)
    m_new = W_new.shape[1]
    nnz_after = (W_new.abs() > WEIGHT_THRESHOLD).sum().item()

    print(f"\n--- FPE intervention at step {split_step} ---")
    print(f"  Hidden Layer Neurons: {m_old} -> {m_new} ({n_split} neurons split)")
    
    if m_old > 0:
        percent_split = (n_split / m_old) * 100
        print(f"  Percentage of neurons split in Layer 1 (Input Weights W): {percent_split:.2f}%")
        print(f"  Percentage of neurons split in Layer 2 (Output Weights W.T): {percent_split:.2f}%")
        print(f"  (Note: In this tied-weight architecture, splitting a hidden neuron inherently splits its symmetric input and output weights simultaneously)")

    print(f"  Non-zeros: {nnz_before} -> {nnz_after} (sparsity preserved)")

    # Build expanded model
    # Build expanded model
    new_model = FeatureRecoveryWithHidden(args.n, m_new, W=W_new, b=model.b.detach()).to(device)
    new_parameter_groups = [
        {"params": new_model.W, "weight_decay": -1.0},
        {"params": new_model.b, "weight_decay": 0.0},
    ]
    new_optimizer = AdamW(new_parameter_groups, lr=1e-2)

    # Initialize optimizer state dictionaries with a dummy step
    new_optimizer.zero_grad()
    dummy_loss = new_model(torch.zeros(1, args.n, device=device)).sum()
    dummy_loss.backward()
    new_optimizer.step()

    # Carefully map the momentum from the old W to the split W
    if len(optimizer.state) > 0:
        old_state_W = optimizer.state[model.W]
        new_state_W = new_optimizer.state[new_model.W]
        
        if 'exp_avg' in old_state_W:
            # Apply the exact same splitting logic to the momentum tensors
            new_state_W['exp_avg'], _ = split_polysemantic_neurons(old_state_W['exp_avg'], reference_W=W_old)
            new_state_W['exp_avg_sq'], _ = split_polysemantic_neurons(old_state_W['exp_avg_sq'], reference_W=W_old)
            new_state_W['step'] = old_state_W['step']
            
            # Copy biases state directly (b shape doesn't change, no split needed)
            new_optimizer.state[new_model.b]['exp_avg'] = optimizer.state[model.b]['exp_avg'].clone()
            new_optimizer.state[new_model.b]['exp_avg_sq'] = optimizer.state[model.b]['exp_avg_sq'].clone()
            new_optimizer.state[new_model.b]['step'] = optimizer.state[model.b]['step']

    # Safely overwrite the old model and optimizer
    model = new_model
    optimizer = new_optimizer

    # Create a mask to enforce sparsity during Phase 2
    sparsity_mask = (model.W.abs() > 0).float()

    # === AFTER SPLIT (immediately after, no training step) ===
    with torch.no_grad():
        x = (
            (torch.rand(args.batch_size, args.n, device=device) < prob)
            * torch.rand(args.batch_size, args.n, device=device)
            * 2
        )
        y, hidden = model(x, return_hidden=True)
        loss_after = criteria(y, x).item()
        pr_after = compute_pr_dim(hidden).item()

    delta_log["after_split"] = {"loss": loss_after, "pr_dim": pr_after, "step": split_step}
    losses.append(loss_after)
    pr_dims.append(pr_after)
    print(f"  [After FPE] step {split_step} | loss: {loss_after:.4e} | D_PR: {pr_after:.2f}")

    # === Phase 2: Resume training for 2000 steps ===
    print(f"\n--- Phase 2: Post-FPE training (steps {split_step+1}–{n_steps_total}) ---")

    for step in range(split_step, n_steps_total):
        x = (
            (torch.rand(args.batch_size, args.n, device=device) < prob)
            * torch.rand(args.batch_size, args.n, device=device)
            * 2
        )

        lr = get_lr(step, 1e-2, n_steps_total)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        y, hidden = model(x, return_hidden=True)
        loss = criteria(y, x)
        loss.backward()

        # Enforce sparsity!
        model.W.grad.data *= sparsity_mask

        optimizer.step()

        with torch.no_grad():
            pr_dim = compute_pr_dim(hidden.detach()).item()

        losses.append(loss.item())
        pr_dims.append(pr_dim)

        if (step + 1) % args.log_interval == 0:
            print(
                f"  step {step + 1}/{n_steps_total} | loss: {loss.item():.4e} | D_PR: {pr_dim:.2f}"
            )

    # === END OF TRAINING ===
    delta_log["end"] = {
        "loss": losses[-1],
        "pr_dim": pr_dims[-1],
        "step": n_steps_total - 1,
    }
    
    final_nnz = (model.W.abs() > WEIGHT_THRESHOLD).sum().item()
    print(f"\n  [End] step {n_steps_total} | loss: {losses[-1]:.4e} | D_PR: {pr_dims[-1]:.2f} | Final Non-zeros: {final_nnz}")

    # Summary
    print("\n--- Delta summary ---")
    b = delta_log["before_split"]
    a = delta_log["after_split"]
    e = delta_log["end"]
    print(f"  Before split:  loss={b['loss']:.4e}, D_PR={b['pr_dim']:.2f}")
    print(f"  After split:   loss={a['loss']:.4e}, D_PR={a['pr_dim']:.2f}  (Δloss={a['loss']-b['loss']:+.4e}, ΔD_PR={a['pr_dim']-b['pr_dim']:+.2f})")
    print(f"  End of train:  loss={e['loss']:.4e}, D_PR={e['pr_dim']:.2f}  (Δloss={e['loss']-a['loss']:+.4e}, ΔD_PR={e['pr_dim']-a['pr_dim']:+.2f})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "losses": losses,
            "pr_dims": pr_dims,
            "delta_log": delta_log,
            "m_before": m_old,
            "m_after": m_new,
        },
        args.output,
    )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

"""
Toy model training with Participation Ratio (PR) dimension logging.

Trains FeatureRecovery in both strong superposition (weight_decay=-1) and weak
superposition (weight_decay=1) regimes, logging loss and PR dimension to verify
whether heavily superposed models exhibit heavy-tailed eigenspectra and lower
intrinsic dimensionality (Recanatesi et al., 2019).
"""

import torch
from torch import nn
import math
import argparse
from adamw import AdamW
from pr_dim import compute_pr_dim


class FeatureRecoveryWithHidden(nn.Module):
    """FeatureRecovery that exposes hidden layer activations for PR dimension computation."""

    def __init__(self, n, m):
        super(FeatureRecoveryWithHidden, self).__init__()
        self.n = n
        self.m = m
        self.W = nn.Parameter(torch.randn(n, m) / math.sqrt(m))
        self.b = nn.Parameter(torch.randn(n))
        self.relu = nn.ReLU()

    def forward(self, x, return_hidden=False):
        # x [batch_size, n]
        hidden = x @ self.W  # [batch_size, m] - hidden layer activations (pre-ReLU)
        output = self.relu(hidden @ self.W.T + self.b)
        if return_hidden:
            return output, hidden
        return output


def get_lr(step, lr, n_steps, warmup_steps=2000):
    step = step + 1
    min_lr = 0.05 * lr
    if warmup_steps < n_steps:
        if step < warmup_steps:
            return lr * step / warmup_steps
        else:
            return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (n_steps - warmup_steps))) + min_lr
    else:
        return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / n_steps)) + min_lr


def train_regime(regime_name, weight_decay, n, m, prob, batch_size, n_steps, log_interval, device):
    """Train a single model and return losses and PR dimensions."""
    model = FeatureRecoveryWithHidden(n, m).to(device)
    prob = prob.to(device)
    parameter_groups = [
        {"params": model.W, "weight_decay": weight_decay},
        {"params": model.b, "weight_decay": 0.0},
    ]
    optimizer = AdamW(parameter_groups, lr=1e-2)
    criteria = nn.MSELoss()

    losses = []
    pr_dims = []

    for step in range(n_steps):
        # Generate data
        x = (torch.rand(batch_size, n, device=device) < prob).float() * torch.rand(
            batch_size, n, device=device
        ) * 2

        # Update learning rate
        lr = get_lr(step, 1e-2, n_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass with hidden activations
        optimizer.zero_grad()
        y, hidden = model(x, return_hidden=True)
        loss = criteria(y, x)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Compute PR dimension (no grad)
        with torch.no_grad():
            pr_dim = compute_pr_dim(hidden.detach())
            pr_dims.append(pr_dim.item())

        if (step + 1) % log_interval == 0:
            print(
                f"  [{regime_name}] step {step + 1}/{n_steps} | loss: {loss.item():.4e} | D_PR: {pr_dim.item():.2f}"
            )

    return losses, pr_dims


def main():
    parser = argparse.ArgumentParser(description="Train toy model with PR dimension logging")
    parser.add_argument("--n", type=int, default=1000, help="number of features")
    parser.add_argument("--m", type=int, default=50, help="hidden dimension")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument("--n_steps", type=int, default=5000, help="training steps")
    parser.add_argument("--log_interval", type=int, default=500, help="log every N steps")
    parser.add_argument("--alpha", type=float, default=0.0, help="feature distribution power-law exponent")
    parser.add_argument("--output", type=str, default="../outputs/exp_pr_dim.pt", help="output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Feature distribution
    prob = torch.tensor([1.0 / i ** (1 + args.alpha) for i in range(1, args.n + 1)])
    prob = prob / prob.sum()

    results = {}

    # Strong superposition: weight_decay = -1 (weight growth)
    print("\n--- Strong superposition (weight_decay=-1) ---")
    losses_strong, pr_dims_strong = train_regime(
        regime_name="strong",
        weight_decay=-1.0,
        n=args.n,
        m=args.m,
        prob=prob,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        log_interval=args.log_interval,
        device=device,
    )
    results["strong"] = {
        "losses": losses_strong,
        "pr_dims": pr_dims_strong,
        "weight_decay": -1.0,
    }

    # Weak superposition: weight_decay = 1
    print("\n--- Weak superposition (weight_decay=1) ---")
    losses_weak, pr_dims_weak = train_regime(
        regime_name="weak",
        weight_decay=1.0,
        n=args.n,
        m=args.m,
        prob=prob,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        log_interval=args.log_interval,
        device=device,
    )
    results["weak"] = {
        "losses": losses_weak,
        "pr_dims": pr_dims_weak,
        "weight_decay": 1.0,
    }

    # Summary
    print("\n--- Summary ---")
    print(
        f"Strong superposition: final loss={losses_strong[-1]:.4e}, final D_PR={pr_dims_strong[-1]:.2f}"
    )
    print(
        f"Weak superposition:   final loss={losses_weak[-1]:.4e}, final D_PR={pr_dims_weak[-1]:.2f}"
    )

    # Save results
    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "strong": results["strong"],
            "weak": results["weak"],
        },
        args.output,
    )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

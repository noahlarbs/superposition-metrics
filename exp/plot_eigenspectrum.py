import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
import argparse
from adamw import AdamW

class FeatureRecoveryWithHidden(nn.Module):
    """Ported model for training feature recovery with access to hidden layer."""
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

def get_lr(step, lr, n_steps, warmup_steps=1000):
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

def train_model(device, n, m, weight_decay, args):
    model = FeatureRecoveryWithHidden(n, m).to(device)
    parameter_groups = [
        {"params": model.W, "weight_decay": weight_decay},
        {"params": model.b, "weight_decay": 0.0},
    ]
    optimizer = AdamW(parameter_groups, lr=1e-2)
    criteria = nn.MSELoss()

    prob = torch.tensor([1.0 / i ** (1 + args.alpha) for i in range(1, n + 1)])
    prob = prob / prob.sum()
    prob = prob.to(device)

    model.train()
    for step in range(args.n_steps):
        x = ((torch.rand(args.batch_size, n, device=device) < prob) * torch.rand(args.batch_size, n, device=device) * 2)
        lr = get_lr(step, 1e-2, args.n_steps, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        y, _ = model(x, return_hidden=True)
        loss = criteria(y, x)
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{args.n_steps} | loss: {loss.item():.4e}")
            
    return model, prob

def main():
    parser = argparse.ArgumentParser(description="Eigenspectrum of Hidden Activations")
    parser.add_argument("--n", type=int, default=1000, help="number of features")
    parser.add_argument("--m", type=int, default=50, help="hidden dimension")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument("--n_steps", type=int, default=5000, help="training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="warmup steps")
    parser.add_argument("--alpha", type=float, default=0.0, help="feature dist alpha")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Training Strong Superposition Model (weight_decay=-1) ---")
    model_strong, prob = train_model(device, args.n, args.m, -1.0, args)

    print("\n--- Training Weak Superposition Model (weight_decay=1) ---")
    model_weak, _ = train_model(device, args.n, args.m, 1.0, args)

    print("\n--- Extracting Covariance and Eigenvalues ---")
    N_eval = 4096
    x_eval = ((torch.rand(N_eval, args.n, device=device) < prob) * torch.rand(N_eval, args.n, device=device) * 2)

    with torch.no_grad():
        _, hidden_strong = model_strong(x_eval, return_hidden=True)
        _, hidden_weak = model_weak(x_eval, return_hidden=True)

        # Compute covariance matrices Σ = (H^T · H) / (N - 1)
        cov_strong = (hidden_strong.T @ hidden_strong) / (N_eval - 1)
        cov_weak = (hidden_weak.T @ hidden_weak) / (N_eval - 1)

        # Calculate eigenvalues
        eigvals_strong = torch.linalg.eigvalsh(cov_strong)
        eigvals_weak = torch.linalg.eigvalsh(cov_weak)

        # Sort descending
        eigvals_strong, _ = torch.sort(eigvals_strong, descending=True)
        eigvals_weak, _ = torch.sort(eigvals_weak, descending=True)
        
        # Clamp near-zero or negative values due to floating-point errors
        eigvals_strong = torch.clamp(eigvals_strong, min=1e-8)
        eigvals_weak = torch.clamp(eigvals_weak, min=1e-8)

        eigvals_strong = eigvals_strong.cpu().numpy()
        eigvals_weak = eigvals_weak.cpu().numpy()

    print("\n--- Plotting ---")
    plt.figure(figsize=(8, 6))
    
    ranks = range(1, args.m + 1)
    
    plt.plot(ranks, eigvals_strong, label="Strong Superposition (wd=-1)\n(Heavy-tailed)", color="blue", linewidth=2.5)
    plt.plot(ranks, eigvals_weak, label="Weak Superposition (wd=1)", color="orange", linewidth=2.5)
    
    plt.xscale("log")
    plt.yscale("log")
    
    plt.title("Eigenspectrum of Hidden Activations ($\Sigma = H^T H / (N-1)$)")
    plt.xlabel("Rank (Log Scale)")
    plt.ylabel("Eigenvalue Magnitude (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs("../newfigures", exist_ok=True)
    out_path = "../newfigures/eigenspectrum_plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()

import torch
from torch import nn
import math
import argparse
import os
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
        hidden = x @ self.W
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

def train_strong_baseline(n, m, prob, batch_size, n_steps, log_interval, device):
    """Train the strong superposition baseline to heavy-tailed stability."""
    model = FeatureRecoveryWithHidden(n, m).to(device)
    prob = prob.to(device)
    parameter_groups = [
        {"params": model.W, "weight_decay": -1.0},
        {"params": model.b, "weight_decay": 0.0},
    ]
    optimizer = AdamW(parameter_groups, lr=1e-2)
    criteria = nn.MSELoss()

    print("\n--- Training Strong Superposition Baseline (Steps 0–5000) ---")
    model.train()
    for step in range(n_steps):
        x = (torch.rand(batch_size, n, device=device) < prob).float() * torch.rand(batch_size, n, device=device) * 2

        lr = get_lr(step, 1e-2, n_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        y, hidden = model(x, return_hidden=True)
        loss = criteria(y, x)
        loss.backward()
        optimizer.step()

        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                pr_dim = compute_pr_dim(hidden.detach())
            print(f"  step {step + 1}/{n_steps} | loss: {loss.item():.4e} | D_PR: {pr_dim.item():.2f}")

    return model

def prune_unstructured(model, sparsity):
    """
    Apply magnitude-based unstructured weight pruning to model.W.
    Zeros out the weights with the smallest absolute magnitudes.
    Assumes sparsity in [0.0, 1.0)
    """
    W = model.W.detach()
    if sparsity <= 0.0:
        return W.clone() # Nothing to prune
        
    k = max(1, int(W.numel() * sparsity)) # Number of elements to zero out
    
    # Flatten, get absolute values, and find the threshold
    flat_W_abs = W.abs().view(-1)
    threshold = torch.kthvalue(flat_W_abs, k).values.item()
    
    # Create mask (True where we keep the weight)
    mask = W.abs() > threshold
    
    # Apply mask
    pruned_W = W * mask
    return pruned_W

def evaluate_model(model, W_pruned, prob, args, device):
    """Evaluate a specific W weighting without updating the model parameter."""
    criteria = nn.MSELoss()
    model.eval()
    
    N_eval = 4096
    
    with torch.no_grad():
        x = (torch.rand(N_eval, args.n, device=device) < prob).float() * torch.rand(N_eval, args.n, device=device) * 2
        
        # Manually compute forward pass using the pruned W matrix
        # (This avoids having to overwrite model.W explicitly if we want to run sequentially)
        hidden = x @ W_pruned
        output = model.relu(hidden @ W_pruned.T + model.b)
        
        loss = criteria(output, x).item()
        pr_dim = compute_pr_dim(hidden).item()
        
    return loss, pr_dim

def main():
    parser = argparse.ArgumentParser(description="Ablation tests on Magnitude-based Unstructured Pruning")
    parser.add_argument("--n", type=int, default=1000, help="number of features")
    parser.add_argument("--m", type=int, default=50, help="hidden dimension")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument("--n_steps", type=int, default=5000, help="training steps")
    parser.add_argument("--log_interval", type=int, default=1000, help="log every N steps")
    parser.add_argument("--alpha", type=float, default=0.0, help="feature distribution power-law exponent")
    parser.add_argument("--output", type=str, default=None, help="output path. If not provided, dynamically generated.")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"../outputs/exp_pruning_n{args.n}_m{args.m}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    prob = torch.tensor([1.0 / i ** (1 + args.alpha) for i in range(1, args.n + 1)], device=device)
    prob = prob / prob.sum()

    # Step 1: Train Baseline
    model = train_strong_baseline(
        n=args.n, m=args.m, prob=prob,
        batch_size=args.batch_size, n_steps=args.n_steps,
        log_interval=args.log_interval, device=device
    )

    # Step 2: Sparsity Ablation Test
    sparsities = [0.0, 0.2, 0.5, 0.75, 0.90, 0.95, 0.99]
    pruning_log = {}

    print("\n--- Ablation Test: Unstructured Pruning ---")
    total_weights = model.m * model.n
    
    for s in sparsities:
        # Generate the masked target weight matrix for this sparsity
        W_pruned = prune_unstructured(model, s)
        
        # Calculate exactly how many weights actually remain
        non_zeros = (W_pruned.abs() > 1e-10).sum().item()
        frac_remaining = non_zeros / total_weights
        
        # Evaluate model with these pruned weights
        loss, pr_dim = evaluate_model(model, W_pruned, prob, args, device)
        
        pruning_log[s] = {
            "target_sparsity": s,
            "non_zeros": non_zeros,
            "frac_remaining": frac_remaining,
            "loss": loss,
            "pr_dim": pr_dim
        }
        
        print(f"  Target Sparsity: {s*100:0.0f}% | Remaining Weights: {frac_remaining*100:0.1f}% ({non_zeros}/{total_weights}) | Loss: {loss:.4e} | D_PR: {pr_dim:.2f}")

    # Step 3: Save Results
    results = {
        "args": vars(args),
        "pruning_log": pruning_log,
        "sparsities": sparsities
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(results, args.output)
    print(f"\nPruning ablation results saved to {args.output}")

if __name__ == "__main__":
    main()

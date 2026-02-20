import torch
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot Pruning Ablation Results")
    parser.add_argument("--data_path", type=str, default="../outputs/exp_pruning_n1000_m50.pt", help="Path to the .pt file to plot")
    args_cmd = parser.parse_args()

    data_path = args_cmd.data_path
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Please provide a path using --data_path if it differs.")
        return

    # Load data
    data = torch.load(data_path, map_location="cpu")
    pruning_log = data["pruning_log"]
    sparsities = data["sparsities"]

    # Extract metrics into lists matching the sorted sparsities
    sparsities_sorted = sorted(sparsities)
    percentages = [s * 100 for s in sparsities_sorted]
    
    losses = [pruning_log[s]["loss"] for s in sparsities_sorted]
    pr_dims = [pruning_log[s]["pr_dim"] for s in sparsities_sorted]

    # Baseline performance (0.0 sparsity)
    if 0.0 in pruning_log:
        baseline_loss = pruning_log[0.0]["loss"]
        baseline_pr_dim = pruning_log[0.0]["pr_dim"]
    else:
        # Fallback if 0.0 isn't explicitly there, though it should be
        baseline_loss = losses[0]
        baseline_pr_dim = pr_dims[0]

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left Plot: Loss Degradation vs. Sparsity
    ax1.plot(percentages, losses, marker='o', color='blue', linewidth=2, label="MSE Loss")
    ax1.axhline(y=baseline_loss, color='black', linestyle=':', linewidth=2, label=f"Baseline Loss ({baseline_loss:.2e})")
    ax1.set_yscale("log")
    ax1.set_title("Loss Degradation vs. Sparsity")
    ax1.set_xlabel("Sparsity (%)")
    ax1.set_ylabel("MSE Loss (Log Scale)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Right Plot: Intrinsic Dimensionality vs. Sparsity
    ax2.plot(percentages, pr_dims, marker='o', color='green', linewidth=2, label="$D_{PR}$")
    ax2.axhline(y=baseline_pr_dim, color='black', linestyle=':', linewidth=2, label=f"Baseline $D_{{PR}}$ ({baseline_pr_dim:.2f})")
    ax2.set_title("Intrinsic Dimensionality vs. Sparsity")
    ax2.set_xlabel("Sparsity (%)")
    ax2.set_ylabel("Participation Ratio ($D_{PR}$)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.suptitle("Holographic Hypothesis: Resilience against Weight Pruning", fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    os.makedirs("../newfigures", exist_ok=True)
    out_path = "../newfigures/pruning_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    
    plt.show()

if __name__ == "__main__":
    main()

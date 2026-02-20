import torch
import matplotlib.pyplot as plt
import os

import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot FPE Experiment Results")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the .pt file to plot")
    args_cmd = parser.parse_args()

    if args_cmd.data_path is None:
        # Default to a generic pattern or require specifying
        data_path = "../outputs/exp_fpe.pt" # We'll leave as default or perhaps just prompt user
    else:
        data_path = args_cmd.data_path

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Please provide a path using --data_path")
        return

    data = torch.load(data_path, map_location="cpu")
    args = data["args"]
    losses = data["losses"]
    pr_dims = data["pr_dims"]
    m_before = data["m_before"]
    m_after = data["m_after"]

    split_point = args["n_steps_pre"]
    total_steps = len(losses)
    steps = range(total_steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left Plot: Loss Landscape
    ax1.plot(steps, losses, label="Loss", color="blue")
    ax1.set_yscale("log")
    ax1.axvline(x=split_point, color="black", linestyle="--", linewidth=2, label="FPE Intervention")
    ax1.set_title("Training Loss Landscape")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right Plot: Intrinsic Dimensionality
    ax2.plot(steps, pr_dims, label="Participation Ratio ($D_{PR}$)", color="green")
    ax2.axvline(x=split_point, color="black", linestyle="--", linewidth=2, label="FPE Intervention")
    
    # Horizontal dotted red lines for ambient dimension ceilings
    ax2.plot([0, split_point], [m_before, m_before], color="red", linestyle=":", linewidth=2, label=f"m = {m_before} (Ceiling)")
    ax2.plot([split_point, total_steps], [m_after, m_after], color="red", linestyle=":", linewidth=2, label=f"m = {m_after} (Ceiling)")
    
    ax2.set_title("Intrinsic Dimensionality ($D_{PR}$)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Participation Ratio ($D_{PR}$)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs("../newfigures", exist_ok=True)
    plt.savefig("../newfigures/fpe_experiment_results.png", dpi=300)
    print("Plot saved to ../newfigures/fpe_experiment_results.png")
    plt.show()

if __name__ == "__main__":
    main()

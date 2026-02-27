import torch
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to .pt data file")
    args = parser.parse_args()

    data = torch.load(args.data_path)
    print(f"Loaded data from {args.data_path}")
    
    losses_full = data["losses_full"]
    pr_dims_full = data["pr_dims_full"]
    losses_quant = data["losses_quant"]
    pr_dims_quant = data["pr_dims_quant"]
    losses_baseline = data["losses_baseline"]
    pr_dims_baseline = data["pr_dims_baseline"]
    
    split_step = data["args"]["n_steps_pre"]
    total_steps = split_step + data["args"]["n_steps_post"]
    
    steps_full = range(len(losses_full))
    steps_quant = range(len(losses_quant))
    steps_baseline = range(len(losses_baseline))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Loss
    ax1.plot(steps_full, losses_full, label="FPE (Full Precision)")
    ax1.plot(steps_quant, losses_quant, label="FPE (BitNet 1.58b Quantized)")
    ax1.plot(steps_baseline, losses_baseline, label="Baseline (No FPE)")
    ax1.axvline(x=split_step, color='r', linestyle='--', label='FPE Intervention')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.legend()
    
    # Plot PR Dimension
    ax2.plot(steps_full, pr_dims_full, label="FPE (Full Precision)")
    ax2.plot(steps_quant, pr_dims_quant, label="FPE (BitNet 1.58b Quantized)")
    ax2.plot(steps_baseline, pr_dims_baseline, label="Baseline (No FPE)")
    ax2.axvline(x=split_step, color='r', linestyle='--', label='FPE Intervention')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Participation Ratio ($D_{PR}$)')
    ax2.set_title('Intrinsic Dimensionality')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs("../newfigures", exist_ok=True)
    save_path = "../newfigures/fpe_quantized_results.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()

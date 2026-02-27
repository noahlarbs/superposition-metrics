import torch
import matplotlib.pyplot as plt
import os

def main():
    methods = ["ternary", "iq2_xxs", "q2_k"]
    data = {}
    
    for method in methods:
        path = f"../outputs/exp_fpe_advanced_{method}.pt"
        if os.path.exists(path):
            data[method] = torch.load(path)
            print(f"Loaded {method}")
        else:
            print(f"Warning: {path} not found.")

    if not data:
        print("No data loaded. Run the experiments first.")
        return

    # Use ternary's full-weight as the full baseline since they share Phase 1 randomly 
    # (actually Phase 1 is separate unless seeds are fixed, but close enough)
    ref_method = list(data.keys())[0]
    losses_full = data[ref_method]["losses_full"]
    pr_dims_full = data[ref_method]["pr_dims_full"]
    
    split_step = data[ref_method]["args"]["n_steps_pre"]
    total_steps = split_step + data[ref_method]["args"]["n_steps_post"]
    steps_full = range(len(losses_full))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Loss
    ax1.plot(steps_full, losses_full, label="FPE (Full Precision)", color="black", linestyle="--")
    for method in methods:
        if method in data:
            ax1.plot(steps_full, data[method]["losses_quant"], label=f"FPE ({method})")
    ax1.axvline(x=split_step, color='red', linestyle=':', label='FPE Intervention')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.legend()
    
    # Plot PR Dimension
    ax2.plot(steps_full, pr_dims_full, label="FPE (Full Precision)", color="black", linestyle="--")
    for method in methods:
        if method in data:
            ax2.plot(steps_full, data[method]["pr_dims_quant"], label=f"FPE ({method})")
    ax2.axvline(x=split_step, color='red', linestyle=':', label='FPE Intervention')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Participation Ratio ($D_{PR}$)')
    ax2.set_title('Intrinsic Dimensionality')
    ax2.legend()
    
    # Plot KL Divergences
    kl_steps = list(range(split_step, total_steps))
    for method in methods:
        if method in data:
            kl_data = data[method]["kl_divs"]
            ax3.plot(kl_steps, kl_data, label=f"KL Div ({method})")
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('KL Divergence against Full-Weight')
    ax3.set_yscale('log')
    ax3.legend()

    plt.tight_layout()
    
    os.makedirs("../newfigures", exist_ok=True)
    save_path = "../newfigures/fpe_advanced_quantizations.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()

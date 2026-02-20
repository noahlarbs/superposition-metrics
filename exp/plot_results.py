import torch
import matplotlib.pyplot as plt
import os

# Load the saved results
data_path = "../outputs/exp_pr_dim.pt"
if not os.path.exists(data_path):
    print(f"Error: Could not find {data_path}. Did you run exp_pr_dim.py first?")
    exit()

data = torch.load(data_path, weights_only=True)
strong = data["strong"]
weak = data["weak"]
n_steps = data["args"]["n_steps"]

# Create x-axis (steps)
steps = list(range(1, n_steps + 1))

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss Landscape
ax1.plot(steps, strong["losses"], label="Strong (Weight Decay = -1)", alpha=0.8)
ax1.plot(steps, weak["losses"], label="Weak (Weight Decay = 1)", alpha=0.8)
ax1.set_yscale("log")
ax1.set_title("Training Loss")
ax1.set_xlabel("Steps")
ax1.set_ylabel("MSE Loss (Log Scale)")
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.5)

# Plot 2: Participation Ratio (PR) Dimension
ax2.plot(steps, strong["pr_dims"], label="Strong Superposition", color="blue")
ax2.plot(steps, weak["pr_dims"], label="Weak Superposition", color="orange")
ax2.axhline(y=data["args"]["m"], color='r', linestyle=':', label=f"Max Dim (m={data['args']['m']})")
ax2.set_title("Intrinsic Dimensionality ($D_{PR}$)")
ax2.set_xlabel("Steps")
ax2.set_ylabel("PR Dimension")
ax2.legend()
ax2.grid(True, ls="--", alpha=0.5)

plt.tight_layout()

# Save to newfigures folder (project root / newfigures)
script_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(script_dir, "..", "newfigures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "pr_dim_loss_landscape.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {fig_path}")

plt.show()

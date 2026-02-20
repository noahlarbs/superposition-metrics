"""
Participation Ratio (PR) dimension for intrinsic dimensionality estimation.

From Recanatesi et al. (2019) "Intrinsic dimension of data representations in deep neural networks":
D_PR = Tr(Σ²) / (Tr(Σ))² = (∑ᵢ λᵢ²) / (∑ᵢ λᵢ)²

Measures the concentration of the eigenvalue distribution. Heavily superposed models
exhibit heavy-tailed eigenspectra and lower intrinsic dimensionality.
"""

import torch

def compute_pr_dim(activations: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the Participation Ratio (PR) dimension from hidden layer activations.
    Formula: D_PR = (Tr(Σ))^2 / Tr(Σ^2)
    """
    # Center the activations
    H = activations - activations.mean(dim=0, keepdim=True)

    batch_size, hidden_dim = H.shape
    n = batch_size - 1  # Bessel's correction for unbiased covariance
    if n <= 0:
        return torch.tensor(float("nan"), device=activations.device, dtype=activations.dtype)

    # Calculate standard covariance matrix Σ: [hidden_dim, hidden_dim]
    # Since hidden_dim (50) << batch_size (2048), this is highly space-efficient.
    sigma = (H.T @ H) / n
    
    # Calculate traces
    tr_sigma = sigma.trace()
    tr_sigma_sq = (sigma @ sigma).trace()
    
    # D_PR = (Tr(Σ))^2 / Tr(Σ^2)
    d_pr = (tr_sigma * tr_sigma) / (tr_sigma_sq + eps)
    
    return d_pr
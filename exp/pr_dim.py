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

    Args:
        activations: Tensor of shape [batch_size, hidden_dim] - hidden layer activations
        eps: Small constant for numerical stability when Tr(Σ) is near zero

    Returns:
        D_PR: Scalar tensor, the participation ratio dimension
    """
    # Center the activations
    H = activations - activations.mean(dim=0, keepdim=True)

    batch_size, hidden_dim = H.shape
    n = batch_size - 1  # Bessel's correction for unbiased covariance
    if n <= 0:
        return torch.tensor(float("nan"), device=activations.device, dtype=activations.dtype)

    # Covariance Σ = (1/(n)) * H.T @ H  [hidden_dim, hidden_dim]
    # Use Gram matrix trick for efficiency when batch_size < hidden_dim:
    # Tr(Σ) = Tr(H.T @ H) / n = ||H||_F² / n
    # Tr(Σ²) = Tr((H.T @ H)²) / n² = Tr((H @ H.T)²) / n² = ||H @ H.T||_F² / n²
    # So D_PR = Tr(Σ²) / (Tr(Σ))² = ||K||_F² / (Tr(K))² where K = H @ H.T

    K = H @ H.T  # [batch_size, batch_size]
    tr_sigma = K.trace() / n
    tr_sigma_sq = (K * K).sum() / (n * n)  # ||K||_F² = Tr(K²)

    d_pr = tr_sigma_sq / (tr_sigma * tr_sigma + eps)
    return d_pr

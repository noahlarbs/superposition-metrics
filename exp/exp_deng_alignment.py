import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import argparse
import time

# ======================================================================
# QUANTIZATION OPS (STE Base)
# ======================================================================
def quantize_w8(w):
    scale = 127.0 / w.abs().max().clamp(min=1e-8)
    w_q = torch.round(w * scale).clamp(-128, 127) / scale
    return w + (w_q - w).detach()

def quantize_w4(w):
    scale = 7.0 / w.abs().max().clamp(min=1e-8)
    w_q = torch.round(w * scale).clamp(-8, 7) / scale
    return w + (w_q - w).detach()

def quantize_w2(w):
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-1, 1) * scale
    return w + (w_q - w).detach()

# ======================================================================
# DENG ET AL. MATHEMATICAL ALIGNMENT METRICS
# ======================================================================
def compute_projection_matrix(A, k):
    """
    Computes projection matrix P onto the top-k eigenspace of A.
    A is approximated here by the uncentered Empirical Fisher F = (1/N) * sum(g_i @ g_i^T)
    """
    if A.shape[0] > 1000:
        _, _, V = torch.svd_lowrank(A, q=k)
        U_k = V
    else:
        L, V = torch.linalg.eigh(A)
        idx = torch.argsort(L, descending=True)
        V = V[:, idx]
        U_k = V[:, :k]
    
    P = U_k @ U_k.T
    return P

def analyze_steady_state(A, k):
    """
    Calculates the Theoretical E_Alignment and the spectral gap m = lambda_{k-1}/lambda_k.
    """
    L, _ = torch.linalg.eigh(A)
    idx = torch.argsort(L, descending=True)
    L = L[idx]
    
    if k < len(L):
        val_k = max(L[k].item(), 1e-12)
        m = L[k-1].item() / val_k
    else:
        m = 1.0
        
    tr_top_k = L[:k].sum().item()
    tr_all = L.sum().item()
    
    e_alignment = tr_top_k / max(tr_all, 1e-12)
    return e_alignment, m, L.detach().cpu().numpy()

def analyze_phase_decay(alignments, start_idx, end_idx):
    """
    Fits exponential decay to the continuous alignment phase: ln(AL_t) = w*t + b.
    """
    slc = alignments[start_idx:end_idx]
    if len(slc) < 2:
        return 0, 0
        
    y = np.log(np.maximum(np.array(slc), 1e-12))
    x = np.arange(len(y)).reshape(-1, 1)
    
    reg = LinearRegression().fit(x, y)
    w = reg.coef_[0]
    r_sq = reg.score(x, y)
    return w, r_sq

# ======================================================================
# TOY MODELS
# ======================================================================
class ToyMatrixModel(nn.Module):
    def __init__(self, d_in, d_hidden, quant_type='W8A16'):
        super().__init__()
        self.d_hidden = d_hidden
        self.quant_type = quant_type
        self.W_in = nn.Parameter(torch.randn(d_in, d_hidden) / np.sqrt(d_in))
        self.W_out = nn.Parameter(torch.randn(d_hidden, d_in) / np.sqrt(d_hidden))
        
    def _quantize(self, w):
        if self.quant_type == 'W8A16': return quantize_w8(w)
        if self.quant_type == 'W4A8': return quantize_w4(w)
        if self.quant_type == 'Ternary': return quantize_w2(w)
        return w
        
    def forward(self, x):
        W_in_sim = self._quantize(self.W_in)
        W_out_sim = self._quantize(self.W_out)
        h = F.relu(x @ W_in_sim)
        return h @ W_out_sim

class ToyL3Model(nn.Module):
    def __init__(self, d_model, d_ff, quant_type='W8A16'):
        super().__init__()
        self.d_ff = d_ff
        self.quant_type = quant_type
        
        self.W_K = nn.Parameter(torch.randn(d_model, 256) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(d_model, 256) / np.sqrt(d_model))
        self.W_in = nn.Parameter(torch.randn(d_model, d_ff) / np.sqrt(d_model)) # w_up
        self.W_out = nn.Parameter(torch.randn(d_ff, d_model) / np.sqrt(d_ff)) # w_mix
        
    def _quantize(self, w):
        if self.quant_type == 'W8A16': return quantize_w8(w)
        if self.quant_type == 'W4A8': return quantize_w4(w)
        if self.quant_type == 'Ternary': return quantize_w2(w)
        return w
        
    def forward(self, x, indices):
        b, t, d = x.shape
        W_in_sim = self._quantize(self.W_in)
        W_out_sim = self._quantize(self.W_out)
        W_K_sim = self._quantize(self.W_K)
        W_V_sim = self._quantize(self.W_V)
        
        local_K = F.embedding(indices, W_K_sim.T)
        local_V = F.embedding(indices, W_V_sim.T)
        
        score = torch.einsum('btd,btsd->bts', x, local_K)
        probs = torch.softmax(score, dim=-1)
        comb_embs = torch.einsum('bts,btsd->btd', probs, local_V)
        
        hidden = comb_embs @ W_in_sim 
        return hidden @ W_out_sim

# ======================================================================
# EXPERIMENT 1: FPE Escape Verification
# ======================================================================
def run_exp1_fpe_escape(device, target_k=5):
    print("\n--- EXPERIMENT 1: FPE Escape Verification ---")
    d_in, d_hidden = 64, 32
    n_steps = 1000
    fpe_step = 600
    
    model = ToyMatrixModel(d_in, d_hidden, quant_type='W4A8').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    X = torch.randn(1024, d_in, device=device)
    Y = X @ (torch.randn(d_in, d_in, device=device)*0.5) 
    
    alignments = []
    
    for step in range(n_steps):
        idx = torch.randint(0, 1024, (128,))
        x_b, y_b = X[idx], Y[idx]
        
        y_hat = model(x_b)
        loss = F.mse_loss(y_hat, y_b)
        
        optimizer.zero_grad()
        loss.backward()
        
        g_mat = model.W_in.grad.detach() 
        A = g_mat @ g_mat.T 
        P = compute_projection_matrix(A, target_k)
        
        num = torch.norm(P @ g_mat)**2
        den = torch.norm(g_mat)**2 + 1e-12
        al_t = (num / den).item()
        alignments.append(al_t)
        
        optimizer.step()
        
        if step == fpe_step:
            print(f"[Step {step}] 🎯 Triggering 128-bit Continuous FPE Expansion!")
            old_w_in = model.W_in.detach()
            old_w_out = model.W_out.detach()
            spawn_w_in = torch.randn(d_in, d_hidden, device=device) * 0.01
            spawn_w_out = torch.randn(d_hidden, d_in, device=device) * 0.01
            
            model.W_in = nn.Parameter(torch.cat([old_w_in, spawn_w_in], dim=1))
            model.W_out = nn.Parameter(torch.cat([old_w_out, spawn_w_out], dim=0))
            model.d_hidden *= 2
            
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    decay_w, decay_r2 = analyze_phase_decay(alignments, fpe_step+10, fpe_step+110)
    print(f"Decay Rate after FPE Injection: w = {decay_w:.4f} (R^2 = {decay_r2:.2f})")
    
    if decay_w < 0:
        print("✅ SUCCESS: Fast Continuous mathematical escape phase (negative slope) established after 128-bit injection.")
    else:
        print("❌ FAILURE: Network collapsed back into alignment trap.")
        
    return alignments, fpe_step

# ======================================================================
# EXPERIMENT 2: Critical Step Size vs Expansion Factor Collision
# ======================================================================
def run_exp2_eta_collision(device, target_k=5):
    print("\n--- EXPERIMENT 2: Critical Step Size (eta_t*) vs Expansion Factor (alpha) ---")
    alphas = [2, 4, 8]  # Scaling factors
    lr_grid = [0.1, 0.01]  # High (violates eta_t*) vs Low (obeys eta_t*)
    d_in, d_hidden = 64, 32
    
    X = torch.randn(1024, d_in, device=device)
    Y = X @ (torch.randn(d_in, d_in, device=device)*0.5) 
    
    for alpha in alphas:
        for lr in lr_grid:
            model = ToyMatrixModel(d_in, d_hidden, quant_type='W4A8').to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            
            # Pre-train to steady state
            for _ in range(200):
                idx = torch.randint(0, 1024, (128,))
                y_hat = model(X[idx])
                loss = F.mse_loss(y_hat, Y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Expand by alpha
            old_w_in = model.W_in.detach()
            old_w_out = model.W_out.detach()
            growth = d_hidden * (alpha - 1)
            spawn_w_in = torch.randn(d_in, growth, device=device) * 0.01
            spawn_w_out = torch.randn(growth, d_in, device=device) * 0.01
            model.W_in = nn.Parameter(torch.cat([old_w_in, spawn_w_in], dim=1))
            model.W_out = nn.Parameter(torch.cat([old_w_out, spawn_w_out], dim=0))
            
            # Post-expansion tracking with grid search LR
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            alignments = []
            
            for _ in range(50):
                idx = torch.randint(0, 1024, (128,))
                y_hat = model(X[idx])
                loss = F.mse_loss(y_hat, Y[idx])
                optimizer.zero_grad()
                loss.backward()
                
                g_mat = model.W_in.grad.detach() 
                A = g_mat @ g_mat.T 
                P = compute_projection_matrix(A, target_k)
                
                num = torch.norm(P @ g_mat)**2
                den = torch.norm(g_mat)**2 + 1e-12
                alignments.append((num / den).item())
                optimizer.step()
            
            al_spike_max = max(alignments)
            print(f"Alpha {alpha}x | post-FPE LR {lr:4.3f} | Max Alignment Spike: {al_spike_max:.3f}")
            if lr == 0.1 and alpha == 8:
                if al_spike_max > 0.8:
                    print("  --> ⚠️ Massive expansion without LR decay violently violates eta_t* boundary!")

# ======================================================================
# EXPERIMENT 3: Quantization Thresholds vs Spectral Gap (Including L3)
# ======================================================================
def run_exp3_spectral_gap(device):
    print("\n--- EXPERIMENT 3: Quant vs Spectral Gap (m) ---")
    quants = ['W8A16', 'W4A8', 'Ternary']
    d_in, d_hidden = 64, 32
    k = 5
    
    X = torch.randn(256, 16, d_in, device=device)
    indices = torch.randint(0, 256, (256, 16, 4), device=device)
    
    print("\nDense FFN Layers:")
    for q in quants:
        model = ToyMatrixModel(d_in, d_hidden, quant_type=q).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Pre-train
        for _ in range(100):
            optimizer.zero_grad()
            y_hat = model(X[:, 0, :])
            loss = F.mse_loss(y_hat, X[:, 0, :] * 0.5)
            loss.backward()
            optimizer.step()
            
        optimizer.zero_grad()
        loss = F.mse_loss(model(X[:, 0, :]), X[:, 0, :] * 0.5)
        loss.backward()
        g_mat = model.W_in.grad.detach() 
        A = g_mat @ g_mat.T 
        e_align, m, _ = analyze_steady_state(A, k)
        print(f"  Quant: {q:7s} | Spectral Gap (m): {m:8.2f} | E_Alignment Limit: {e_align:.3f}")
        
    print("\nL3 Large Lookup Layers:")
    for q in quants:
        model_l3 = ToyL3Model(d_in, d_hidden, quant_type=q).to(device)
        optimizer = torch.optim.SGD(model_l3.parameters(), lr=0.1)
        
        for _ in range(100):
            optimizer.zero_grad()
            y_hat = model_l3(X, indices)
            loss = F.mse_loss(y_hat, X * 0.5)
            loss.backward()
            optimizer.step()
            
        optimizer.zero_grad()
        loss = F.mse_loss(model_l3(X, indices), X * 0.5)
        loss.backward()
        g_mat = model_l3.W_in.grad.detach() 
        # W_in is effectively d_model x d_up (D x d_ff) here. Reshape for approx Hessian
        A = g_mat @ g_mat.T 
        e_align, m, _ = analyze_steady_state(A, k)
        print(f"  Quant: {q:7s} | Spectral Gap (m): {m:8.2f} | E_Alignment Limit: {e_align:.3f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Phase 6 Mathematical Verifications on {device}")
    
    # Exp 1
    run_exp1_fpe_escape(device)
    
    # Exp 2
    run_exp2_eta_collision(device)
    
    # Exp 3
    run_exp3_spectral_gap(device)

if __name__ == "__main__":
    main()

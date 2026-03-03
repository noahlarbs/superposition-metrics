import sys
class _DL:
    def __init__(self, f):
        self.t=sys.stdout; self.l=open(f, 'w')
    def write(self, m):
        self.t.write(m); self.l.write(m); self.l.flush()
    def flush(self):
        self.t.flush(); self.l.flush()
sys.stdout = _DL('live_output_part7.log')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.subplots as subplots
import matplotlib.pyplot as plt
import os

# ======================================================================
# QUANTIZATION OPS (STE Base)
# ======================================================================
def quantize_w8(w):
    scale = 127.0 / w.abs().max().clamp(min=1e-8)
    w_q = torch.round(w * scale).clamp(-128, 127) / scale
    return w + (w_q - w).detach()

def quantize_a16(x):
    scale = 32767.0 / x.abs().max().clamp(min=1e-8)
    x_q = torch.round(x * scale).clamp(-32768, 32767) / scale
    return x + (x_q - x).detach()

def quantize_w4(w):
    scale = 7.0 / w.abs().max().clamp(min=1e-8)
    w_q = torch.round(w * scale).clamp(-8, 7) / scale
    return w + (w_q - w).detach()

def quantize_a8(x):
    return quantize_w8(x)

def quantize_iq2_xxs(w, block_size=32):
    orig_shape = w.shape
    w_flat = w.view(-1)
    
    pad_len = (block_size - (w_flat.size(0) % block_size)) % block_size
    if pad_len > 0:
        w_padded = torch.cat([w_flat, torch.zeros(pad_len, device=w.device)])
    else:
        w_padded = w_flat
        
    blocks = w_padded.view(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    blocks_norm = blocks / scales
    
    thresholds = torch.tensor([-1.0, 0.0, 1.0], device=w.device)
    diffs = torch.abs(blocks_norm.unsqueeze(-1) - thresholds)
    min_idx = torch.argmin(diffs, dim=-1)
    blocks_q = thresholds[min_idx] * scales
    
    w_q_flat = blocks_q.view(-1)
    if pad_len > 0:
        w_q_flat = w_q_flat[:-pad_len]
        
    w_q = w_q_flat.view(orig_shape)
    return w + (w_q - w).detach()

def compute_projection_matrix(A, k):
    if A.shape[0] > 1000:
        _, _, V = torch.svd_lowrank(A, q=k)
        U_k = V
    else:
        L, V = torch.linalg.eigh(A)
        idx = torch.argsort(L, descending=True)
        V = V[:, idx]
        U_k = V[:, :k]
    return U_k @ U_k.T

class ToyMatrixModel(nn.Module):
    def __init__(self, d_in, d_hidden, quant_type='FP32'):
        super().__init__()
        self.d_hidden = d_hidden
        self.quant_type = quant_type
        self.W_in = nn.Parameter(torch.randn(d_in, d_hidden) / np.sqrt(d_in))
        self.W_out = nn.Parameter(torch.randn(d_hidden, d_in) / np.sqrt(d_hidden))
        
    def _quantize_w(self, w):
        if self.quant_type == 'W8A16': return quantize_w8(w)
        if self.quant_type == 'W4A8': return quantize_w4(w)
        if self.quant_type == 'iq2_xxs': return quantize_iq2_xxs(w)
        return w

    def _quantize_a(self, a):
        if self.quant_type == 'W8A16': return quantize_a16(a)
        if self.quant_type in ['W4A8', 'iq2_xxs']: return quantize_a8(a)
        return a
        
    def forward(self, x):
        W_in_sim = self._quantize_w(self.W_in)
        W_out_sim = self._quantize_w(self.W_out)
        
        x_sim = self._quantize_a(x)
        h = F.relu(x_sim @ W_in_sim)
        h_sim = self._quantize_a(h)
        
        return h_sim @ W_out_sim

# ======================================================================
# EXPERIMENT 4: Targeted FPE Escape Matrix (Part 7)
# ======================================================================
def run_deng_superposition_matrix():
    print("\n--- EXPERIMENT 4 (PART 7): TARGETED NEURON EXPANSION VIA DENG ALIGNMENT TRAP ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Grid Search Parameters
    d_in = 1024
    d_hidden_starts = [64, 128, 256, 512]
    quant_schemes = ['FP32', 'W8A16', 'W4A8', 'iq2_xxs']
    target_k = 100 
    n_steps = 5000
    
    # Common Data
    X = torch.randn(2048, d_in, device=device)
    Y = X @ (torch.randn(d_in, d_in, device=device) * 0.5)
    
    results = {}

    for d_start in d_hidden_starts:
        for q in quant_schemes:
            key = f"D{d_start}_{q}"
            print(f"\n=======================================================")
            print(f"🚀 STARTING DENG MATRIX: Initial Width={d_start} | Quant={q}")
            print(f"=======================================================")
            
            model = ToyMatrixModel(d_in, d_start, quant_type=q).to(device)
            lr = 0.5 if q == 'iq2_xxs' else 0.15
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            
            e_al_target = target_k / model.d_hidden  
            trap_threshold = e_al_target * 1.5 
            
            print(f"Tracking Network Geometry: d_hidden = {model.d_hidden}, Feature Rank = {target_k}")
            print(f"Expected Minimum Alignment (m/d): {e_al_target:.3f}")
            print(f"Triggering expansion when Empirical AL_t > {trap_threshold:.3f} (Superposition Limit)")
            
            fpe_step = None
            expansions = 0
            
            losses = []
            al_ts = []
            trap_lines = []
            flops = []
            acc_flops = 0
            fpe_events = []
            
            for step in range(n_steps):
                idx = torch.randint(0, 2048, (1024,))
                x_b, y_b = X[idx], Y[idx]
                
                y_hat = model(x_b)
                loss = F.mse_loss(y_hat, y_b)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Deng Metric: Spectral Alignment Computation
                g_mat = model.W_in.grad.detach() 
                A = g_mat @ g_mat.T 
                P = compute_projection_matrix(A, target_k)
                
                num = torch.norm(P @ g_mat)**2
                den = torch.norm(g_mat)**2 + 1e-12
                al_t = (num / den).item()
                
                # Cost Heuristic
                step_flops = 2 * 1024 * d_in * model.d_hidden
                acc_flops += step_flops
                
                losses.append(loss.item())
                al_ts.append(al_t)
                trap_lines.append(trap_threshold)
                flops.append(acc_flops)

                if step % 500 == 0:
                    print(f"  [Step {step:4d}] Loss: {loss.item():.4f} | AL_t: {al_t:.3f} | E_AL: {e_al_target:.3f} | Trap Threshold: {trap_threshold:.3f}")

                # Deng Trap Detection (allow multiple expansions but with a cooldown)
                if al_t > trap_threshold and (step - (fpe_step if fpe_step else -100)) > 100:
                    fpe_step = step
                    expansions += 1
                    width_increment = model.d_hidden // 2 # 50% Geometric Detonation
                    
                    print(f"\n  [Step {step}] 🎯 DENG SUPERPOSITION TRAP DETECTED:")
                    print(f"    - Empirical Alignment (AL_t) = {al_t:.3f} > {trap_threshold:.3f}")
                    print(f"    - Executing TARGETED Fractional Expansion to {model.d_hidden + width_increment} neurons!")
                    
                    old_w_in = model.W_in.detach()
                    old_w_out = model.W_out.detach()
                    
                    spawn_w_in = torch.randn(d_in, width_increment, device=device) * 0.01
                    spawn_w_out = torch.randn(width_increment, d_in, device=device) * 0.01
                    
                    model.W_in = nn.Parameter(torch.cat([old_w_in, spawn_w_in], dim=1))
                    model.W_out = nn.Parameter(torch.cat([old_w_out, spawn_w_out], dim=0))
                    
                    fpe_events.append({'flops': acc_flops, 'd_hidden': model.d_hidden + width_increment, 'al_t': al_t})
                    model.d_hidden += width_increment
                    
                    # Recalculate Deng E_AL after expansion
                    e_al_target = target_k / model.d_hidden
                    trap_threshold = e_al_target * 1.5
                    
                    # Stabilize new parameter state
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr*0.8, momentum=0.9)
                    
                optimizer.step()
                
            results[key] = {
                'losses': losses,
                'al_ts': al_ts,
                'trap_lines': trap_lines,
                'flops': flops,
                'fpe_events': fpe_events,
                'final_d_hidden': model.d_hidden,
                'expansions': expansions
            }
            
            if expansions > 0:
                 print(f"  ✅ SUCCESS: Expanded {expansions} times. Final Width: {model.d_hidden}")
            else:
                 print(f"  ❌ FAILURE: Never reached Deng Superposition. Capacity too high.")
                 
    # ---------------------------------------------------------
    # Visualizing the Deng Matrix
    # ---------------------------------------------------------
    fig, axes = plt.subplots(len(d_hidden_starts), len(quant_schemes), figsize=(20, 16))
    fig.suptitle("Experiment 7: Deng Superposition Escape via Adaptive FPE", fontsize=18)
    
    for i, d_start in enumerate(d_hidden_starts):
        for j, q in enumerate(quant_schemes):
            ax = axes[i, j]
            key = f"D{d_start}_{q}"
            res = results[key]
            
            ax.set_title(f"Start Width: {d_start} | {q}")
            ax.set_ylabel("Deng Spectral Alignment ($AL_t$)")
            ax.set_xlabel("Accumulated FLOPS")
            
            # Plot Alignment Trace
            ax.plot(res['flops'], res['al_ts'], color='blue', label='Empirical $AL_t$', alpha=0.8)
            ax.plot(res['flops'], res['trap_lines'], color='red', linestyle='--', label='Trap Threshold ($E_{AL} \\times 1.5$)', alpha=0.9)
            
            # Mark FPE Hits
            for ev in res['fpe_events']:
                ax.axvline(x=ev['flops'], color='orange', linestyle=':', label=f"Expanded to {ev['d_hidden']}")
                ax.scatter(ev['flops'], ev['al_t'], color='red', s=50, zorder=5) # Red dot at trigger point
                
            ax.set_ylim(0, 1.0)
            if i == 0 and j == 0:
                ax.legend(fontsize='small')
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = f"../outputs/experiment7_deng_matrix.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"\nDeng Trap Multi-Scale Map saved to {output_path}")

if __name__ == '__main__':
    run_deng_superposition_matrix()

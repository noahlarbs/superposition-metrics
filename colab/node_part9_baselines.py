import sys
class _DL:
    def __init__(self, f):
        self.t=sys.stdout; self.l=open(f, 'w')
    def write(self, m):
        self.t.write(m); self.l.write(m); self.l.flush()
    def flush(self):
        self.t.flush(); self.l.flush()
sys.stdout = _DL('live_output_part9_baselines.log')

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
import wandb
from datasets import load_dataset
from transformers import GPT2Tokenizer

# ======================================================================
# DATA LOADING (TinyStories)
# ======================================================================
def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def get_lr(step, base_lr, total_steps, warmup_steps=1000):
    min_lr = 0.2 * base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)) + min_lr

# ======================================================================
# FAKE QUANTIZATION OPS (STE)
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

def quantize_ternary(w):
    scale = w.abs().mean().clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-1, 1) * scale
    return w + (w_q - w).detach()

def quantize_q2_k(w, block_size=32):
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
    
    # 4 distinct states = 2 bits
    thresholds = torch.tensor([-1.0, -0.33, 0.33, 1.0], device=w.device)
    diffs = torch.abs(blocks_norm.unsqueeze(-1) - thresholds)
    min_idx = torch.argmin(diffs, dim=-1)
    blocks_q = thresholds[min_idx] * scales
    
    w_q_flat = blocks_q.view(-1)
    if pad_len > 0:
        w_q_flat = w_q_flat[:-pad_len]
        
    w_q = w_q_flat.view(orig_shape)
    return w + (w_q - w).detach()

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
    
    # iq2_xxs drops extreme weights to a ternary distribution with a heavy zero bias
    thresholds = torch.tensor([-1.0, 0.0, 1.0], device=w.device)
    diffs = torch.abs(blocks_norm.unsqueeze(-1) - thresholds)
    min_idx = torch.argmin(diffs, dim=-1)
    blocks_q = thresholds[min_idx] * scales
    
    w_q_flat = blocks_q.view(-1)
    if pad_len > 0:
        w_q_flat = w_q_flat[:-pad_len]
        
    w_q = w_q_flat.view(orig_shape)
    return w + (w_q - w).detach()

# ======================================================================
# STATIC FFN LAYER (NO FPE)
# ======================================================================
class StaticFFN(nn.Module):
    def __init__(self, d_model, d_ff, base_quant):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.base_quant = base_quant
        
        self.W_in = nn.Parameter(torch.randn(d_model, d_ff) / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_ff))
        self.W_out = nn.Parameter(torch.randn(d_ff, d_model) / math.sqrt(d_ff))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def _get_base_quant_fn(self):
        if self.base_quant == 'W8A16': return quantize_w8
        if self.base_quant == 'W4A8': return quantize_w4
        if self.base_quant == 'Ternary': return quantize_ternary
        if self.base_quant == 'q2_k': return quantize_q2_k
        if self.base_quant == 'iq2_xxs': return quantize_iq2_xxs
        
        # Enforce exact floating point container states
        if self.base_quant == 'FP64': return lambda x: x.to(torch.float64)
        if self.base_quant == 'FP32': return lambda x: x.to(torch.float32)
        
        return lambda x: x

    def forward(self, x):
        q_fn = self._get_base_quant_fn()
        
        W_in_sim = q_fn(self.W_in)
        W_out_sim = q_fn(self.W_out)

        # Activation Constraints
        if self.base_quant == 'W8A16':
            x_sim = quantize_a16(x)
        elif self.base_quant in ['W4A8', 'Ternary', 'q2_k', 'iq2_xxs']:
            x_sim = quantize_a8(x)
        else:
            x_sim = x

        hidden = F.gelu(x_sim @ W_in_sim + self.b_in)
        
        if self.base_quant == 'W8A16':
            hidden_sim = quantize_a16(hidden)
        elif self.base_quant in ['W4A8', 'Ternary', 'q2_k', 'iq2_xxs']:
            hidden_sim = quantize_a8(hidden)
        else:
            hidden_sim = hidden
            
        out = hidden_sim @ W_out_sim + self.b_out
        return out

# ======================================================================
# MODEL DEFINITION 
# ======================================================================
class StaticTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=64, base_quant='W8A16'):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = StaticFFN(d_model, d_ff, base_quant)

    def forward(self, x):
        b, seq_len, _ = x.shape
        x_norm = quantize_a16(self.ln1(x))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False, is_causal=True, attn_mask=causal_mask)
        x = x + attn_out
        
        x_norm = self.ln2(x)
        x = x + self.ffn(x_norm)
        return x

class StaticTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=64, base_quant='W8A16'):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(256, d_model) # Context window 256
        
        self.layers = nn.ModuleList([
            StaticTransformerBlock(d_model, n_heads, d_ff, base_quant)
            for _ in range(3)
        ])
        
        self.ln3 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        b, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        
        x = quantize_a16(quantize_w8(self.embed(x))) + quantize_w8(self.pos_embed(pos))
        
        for layer in self.layers:
            x = layer(x)
                
        x = self.ln3(x)
        logits = quantize_w8(self.head.weight) @ quantize_a16(x).transpose(1, 2)
        logits = logits.transpose(1, 2)
        
        return logits

# ======================================================================
# TRAINING LOOP
# ======================================================================
def run_baseline_experiment(base_quant, d_model, d_ff, args, device, train_data, val_data, name_prefix):
    name = f"{name_prefix}_{base_quant}_d{d_model}_ff{d_ff}"
    print(f"\n=======================================================")
    print(f"BASELINE RUN: {name}")
    print(f"=======================================================")
    
    wandb.init(
        project="superposition-metrics",
        name=name,
        config={
            **vars(args),
            "base_quant": base_quant,
            "d_model": d_model,
            "d_ff": d_ff,
            "architecture": "baseline_static"
        }
    )
    
    vocab_size = 50257
    dtype = torch.float64 if base_quant == 'FP64' else torch.float32

    model = StaticTransformer(
        vocab_size, d_model=d_model, n_heads=4, d_ff=d_ff, base_quant=base_quant
    ).to(device=device, dtype=dtype)
    
    base_lr = 5e-3 if base_quant in ['Ternary', 'iq2_xxs', 'q2_k'] else 1e-3
    base_wd = 0.0 if base_quant in ['Ternary', 'iq2_xxs', 'q2_k'] else 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_wd)
    
    steps = args.n_steps
    log_interval = args.log_interval
    batch_size = args.batch_size
    seq_len = 256
    
    losses = []
    proxy_flops = []
    accumulated_flops = 0.0
    
    def get_step_flops():
        return model.d_model * model.layers[0].ffn.d_ff * 2 * batch_size * seq_len * len(model.layers)
    
    for step in range(steps):
        if step % 50 == 0 and os.path.exists('control.flag'):
            with open('control.flag', 'r') as f:
                cmd = f.read().strip().lower()
            if 'skip' in cmd:
                print(f"\n[CONTROL] 🛑 Received 'skip' command! Aborting step {step}...")
                os.remove('control.flag') 
                break
                
        t0 = time.time()
        xb, yb = get_batch(train_data, seq_len, batch_size, device)
        
        lr = get_lr(step, base_lr, steps)
        for pg in optimizer.param_groups:
             pg["lr"] = lr
             
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        loss.backward()
        
        optimizer.step()
        accumulated_flops += get_step_flops()
        
        if step % log_interval == 0:
            with torch.no_grad():
                model.eval()
                xv, yv = get_batch(val_data, seq_len, batch_size, device)
                val_logits = model(xv)
                val_loss = F.cross_entropy(val_logits.reshape(-1, val_logits.size(-1)), yv.reshape(-1)).item()
                model.train()
                
            losses.append(val_loss)
            proxy_flops.append(accumulated_flops)

            if step % (log_interval * 5) == 0:
                 print(f"  Step {step:4d} | Val Loss {val_loss:.4f} | d_ff {d_ff}")

            wandb.log({
                "train/loss": loss.item() if hasattr(loss, 'item') else 0.0,
                "val/loss": val_loss,
                "model/d_ff": d_ff,
                "train/step": step,
                "train/accumulated_flops": accumulated_flops
            })

    wandb.finish()
    return {
        'losses': losses,
        'proxy_flops': proxy_flops,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=50)
    args, _ = parser.parse_known_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading TinyStories dataset (Subset)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    text = " ".join([ds[i]['text'] for i in range(100)])
    tokens = np.array(tokenizer.encode(text))
    
    split_idx = int(0.9 * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]
    
    quants = ['FP64', 'FP32', 'W8A16', 'W4A8', 'Ternary', 'q2_k', 'iq2_xxs']
    
    configs = [
        # Small Baseline (Equivalent to starting width of FPE experiments)
        {"name": "Small", "d_model": 64, "d_ff": 64}, 
        # Mid Baseline
        {"name": "Mid", "d_model": 64, "d_ff": 256},
        # Large Baseline (Equivalent to final capped expansion of FPE experiments)
        {"name": "Large", "d_model": 64, "d_ff": 512} 
    ]
    
    results = {}
    for c in configs:
        for q in quants:
            k = f"{c['name']}_{q}"
            results[k] = run_baseline_experiment(q, c['d_model'], c['d_ff'], args, device, train_data, val_data, c['name'])
            
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Experiment 9: Pure Baseline Models (Precision vs Capacity)", fontsize=18)
    
    for i, c in enumerate(configs):
        ax = axes[i]
        ax.set_title(f"{c['name']} Model (d_ff={c['d_ff']})")
        ax.set_ylabel("Validation Loss")
        ax.set_xlabel("Accumulated FLOPS Proxy")
        
        for q in quants:
            k = f"{c['name']}_{q}"
            res = results[k]
            # Highlight FP64 and FP32 specifically
            if q == 'FP64':
                ax.plot(res['proxy_flops'], res['losses'], label=f"{q}", color='black', linewidth=2.5, zorder=10)
            elif q == 'FP32':
                ax.plot(res['proxy_flops'], res['losses'], label=f"{q}", color='blue', linewidth=2.0, zorder=9)
            else:
                ax.plot(res['proxy_flops'], res['losses'], label=f"{q}", alpha=0.6)
            
        ax.legend()
        
    plt.tight_layout()
    output_path = f"../outputs/experiment9_baselines.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Matrix graph saved to {output_path}")

main()

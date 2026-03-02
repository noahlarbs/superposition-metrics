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

def apply_precision_age(w, age, base_quant_fn):
    if age == 0: 
        return w # FP32
    elif age == 1:
        return w.half().float()  
    elif age == 2:
        return quantize_w8(w)
    else:
        return base_quant_fn(w)

# ======================================================================
# PROGRESSIVE FFN LAYER
# ======================================================================
class ProgressiveExpandableFFN(nn.Module):
    def __init__(self, d_model, d_ff, base_quant, is_ageing):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.base_quant = base_quant
        self.is_ageing = is_ageing
        
        self.W_in = nn.Parameter(torch.randn(d_model, d_ff) / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_ff))
        self.W_out = nn.Parameter(torch.randn(d_ff, d_model) / math.sqrt(d_ff))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        
        # Track generational ages per FFN neuron
        # All initial neurons start at maximum archive age (3.0)
        self.register_buffer("neuron_ages", torch.full((d_ff,), 3.0, dtype=torch.float32))

    def _get_base_quant_fn(self):
        if self.base_quant == 'W8A16': return quantize_w8
        if self.base_quant == 'W4A8': return quantize_w4
        if self.base_quant == 'Ternary': return quantize_ternary
        if self.base_quant == 'q2_k': return quantize_q2_k
        if self.base_quant == 'iq2_xxs': return quantize_iq2_xxs
        return lambda x: x

    def forward(self, x, return_hidden=False):
        q_fn = self._get_base_quant_fn()
        
        W_in_sim = torch.empty_like(self.W_in)
        W_out_sim = torch.empty_like(self.W_out)
        
        for i in range(self.d_ff):
            age = int(self.neuron_ages[i].item())
            if self.is_ageing:
                W_in_sim[:, i] = apply_precision_age(self.W_in[:, i], age, q_fn)
                W_out_sim[i, :] = apply_precision_age(self.W_out[i, :], age, q_fn)
            else:
                if age >= 3:
                     W_in_sim[:, i] = q_fn(self.W_in[:, i])
                     W_out_sim[i, :] = q_fn(self.W_out[i, :])
                else:
                     W_in_sim[:, i] = self.W_in[:, i]
                     W_out_sim[i, :] = self.W_out[i, :]

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
        
        if return_hidden:
            return out, hidden
        return out

# ======================================================================
# L3 LARGE LOOKUP LAYER
# ======================================================================
class ProgressiveL3Layer(nn.Module):
    def __init__(self, d_model, d_ff, base_quant, is_ageing, n_emb=16384, k=128):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff  # Identical to d_up
        self.n_emb = n_emb
        self.k = k # Embeddings per token (paper max was 512 but memory intensive locally)
        self.base_quant = base_quant
        self.is_ageing = is_ageing
        
        # Token mapping fallback table
        vocab_size = 50257
        self.register_buffer("emb_alloc", torch.clamp(torch.arange(0, vocab_size * k, device='cpu') % n_emb, max=n_emb-1).view(vocab_size, k))
        
        # The L3 lookup system (w_k and w_v equivalents)
        self.W_K = nn.Parameter(torch.randn(d_model, n_emb) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(d_model, n_emb) / math.sqrt(d_model))
        
        # Geometrically expandable projection network
        self.W_in = nn.Parameter(torch.randn(d_model, d_ff) / math.sqrt(d_model)) # acts as w_up
        self.W_out = nn.Parameter(torch.randn(d_ff, d_model) / math.sqrt(d_ff)) # acts as top half of w_mix
        self.W_skip = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model)) # acts as bottom half of w_mix
        
        self.b_in = nn.Parameter(torch.zeros(d_ff))    # Unused in strict L3 but needed for detonate wrapper compatibility
        self.b_out = nn.Parameter(torch.zeros(d_model)) 
        
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_ff)
        
        # Track generational ages per FFN neuron in the expanded dim
        self.register_buffer("neuron_ages", torch.full((d_ff,), 3.0, dtype=torch.float32))

    def _get_base_quant_fn(self):
        if self.base_quant == 'W8A16': return quantize_w8
        if self.base_quant == 'W4A8': return quantize_w4
        if self.base_quant == 'Ternary': return quantize_ternary
        if self.base_quant == 'q2_k': return quantize_q2_k
        if self.base_quant == 'iq2_xxs': return quantize_iq2_xxs
        return lambda x: x

    def forward(self, x, input_ids=None, return_hidden=False):
        b, t, d = x.shape
        q_fn = self._get_base_quant_fn()
        
        W_in_sim = torch.empty_like(self.W_in)
        W_out_sim = torch.empty_like(self.W_out)
        
        # FPE Ageing on Projection
        for i in range(self.d_ff):
            age = int(self.neuron_ages[i].item())
            if self.is_ageing:
                W_in_sim[:, i] = apply_precision_age(self.W_in[:, i], age, q_fn)
                W_out_sim[i, :] = apply_precision_age(self.W_out[i, :], age, q_fn)
            else:
                if age >= 3:
                     W_in_sim[:, i] = q_fn(self.W_in[:, i])
                     W_out_sim[i, :] = q_fn(self.W_out[i, :])
                else:
                     W_in_sim[:, i] = self.W_in[:, i]
                     W_out_sim[i, :] = self.W_out[i, :]

        # Full Quantization of L3 Embeddings and Skips
        W_K_sim = q_fn(self.W_K)
        W_V_sim = q_fn(self.W_V)
        W_skip_sim = q_fn(self.W_skip)

        # Activation Constraints
        if self.base_quant == 'W8A16': x_sim = quantize_a16(x)
        elif self.base_quant in ['W4A8', 'Ternary', 'q2_k', 'iq2_xxs']: x_sim = quantize_a8(x)
        else: x_sim = x

        A = self.norm_in(x_sim)
        
        # Pull allocated embedding clusters
        indices = self.emb_alloc[input_ids] # [B, T, K]
        local_K = F.embedding(indices, W_K_sim.T) # [B, T, K, D]
        local_V = F.embedding(indices, W_V_sim.T) # [B, T, K, D]
        
        # Local routing logic
        score = torch.einsum('btd,btsd->bts', A, local_K)
        probs = torch.softmax(score, dim=-1)
        comb_embs = torch.einsum('bts,btsd->btd', probs, local_V)
        
        # L3 projection (w_up) mapped identically to FFN W_in for FPE geometric detonation
        hidden = comb_embs @ W_in_sim 
        
        if self.base_quant == 'W8A16':
            hidden_sim = quantize_a16(hidden)
        elif self.base_quant in ['W4A8', 'Ternary', 'q2_k', 'iq2_xxs']:
            hidden_sim = quantize_a8(hidden)
        else:
            hidden_sim = hidden
            
        # L3 projection (w_mix) mapped identically to FFN W_out with isolated skip connection
        out = self.norm_out(hidden_sim) @ W_out_sim + x_sim @ W_skip_sim
        
        if return_hidden:
            return out, hidden
        return out


# ======================================================================
# MODEL DEFINITION 
# ======================================================================
class ProgressiveTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff_start=64, base_quant='W8A16', is_ageing=True, use_l3=False):
        super().__init__()
        self.use_l3 = use_l3
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        
        if self.use_l3:
            self.ffn = ProgressiveL3Layer(d_model, d_ff_start, base_quant, is_ageing)
        else:
            self.ffn = ProgressiveExpandableFFN(d_model, d_ff_start, base_quant, is_ageing)

    def forward(self, x, input_ids_cache=None, return_hidden=False):
        b, seq_len, _ = x.shape
        x_norm = quantize_a16(self.ln1(x))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False, is_causal=True, attn_mask=causal_mask)
        x = x + attn_out
        
        x_norm = self.ln2(x)
        if return_hidden:
            if self.use_l3:
                ffn_out, hidden_acts = self.ffn(x_norm, input_ids=input_ids_cache, return_hidden=True)
            else:
                ffn_out, hidden_acts = self.ffn(x_norm, return_hidden=True)
            x = x + ffn_out
            return x, hidden_acts
        else:
            if self.use_l3:
                x = x + self.ffn(x_norm, input_ids=input_ids_cache)
            else:
                x = x + self.ffn(x_norm)
            return x

class ProgressiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff_start=64, base_quant='W8A16', is_ageing=True, l3_count=0):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(256, d_model) # Context window 256
        
        self.layers = nn.ModuleList([
            ProgressiveTransformerBlock(d_model, n_heads, d_ff_start, base_quant, is_ageing, use_l3=(i < l3_count))
            for i in range(3)
        ])
        
        self.ln3 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, return_hidden=False):
        b, seq_len = x.shape
        input_ids_cache = x  # Needed to look up L3 specific token routing
        pos = torch.arange(seq_len, device=x.device)
        
        x = quantize_a16(quantize_w8(self.embed(x))) + quantize_w8(self.pos_embed(pos))
        
        all_hidden = []
        for layer in self.layers:
            if return_hidden:
                x, hidden = layer(x, input_ids_cache=input_ids_cache, return_hidden=True)
                all_hidden.append(hidden)
            else:
                x = layer(x, input_ids_cache=input_ids_cache)
                
        x = self.ln3(x)
        logits = quantize_w8(self.head.weight) @ quantize_a16(x).transpose(1, 2)
        logits = logits.transpose(1, 2)
        
        if return_hidden:
            return logits, all_hidden
        return logits

# ======================================================================
# FPE UTILS
# ======================================================================
def compute_effective_rank(hidden_acts):
    """Computes effective rank of the FFN activation covariance matrix"""
    # hidden_acts: [B, Seq, d_ff]
    flat_acts = hidden_acts.reshape(-1, hidden_acts.shape[-1])
    # Subsample for speed
    if flat_acts.shape[0] > 2000:
        idx = torch.randperm(flat_acts.shape[0])[:2000]
        flat_acts = flat_acts[idx]
        
    cov = torch.cov(flat_acts.T)
    _, S, _ = torch.linalg.svd(cov)
    
    p = (S + 1e-12) / (S + 1e-12).sum()
    entropy = -torch.sum(p * torch.log(p))
    effective_rank = torch.exp(entropy)
    return effective_rank.item()


def geometric_detonate_layer(ffn_layer, growth_factor=2):
    """
    Splits the FFN strictly orthogonally and increments ages.
    Newly spawned neurons get Age=0 (FP32).
    """
    device = ffn_layer.W_in.device
    W_in_old = ffn_layer.W_in.detach()
    b_in_old = ffn_layer.b_in.detach()
    W_out_old = ffn_layer.W_out.detach()
    
    d_model, d_ff_old = W_in_old.shape
    d_ff_new = d_ff_old * growth_factor
    n_added = d_ff_new - d_ff_old
    
    # 1. Step ages
    new_ages = ffn_layer.neuron_ages + 1.0
    spawned_ages = torch.zeros(n_added, dtype=torch.float32, device=device)
    ffn_layer.neuron_ages = torch.cat([new_ages, spawned_ages])
    
    # 2. Geometric Splitting
    W_in_spawn = torch.zeros(d_model, n_added, device=device)
    b_in_spawn = torch.zeros(n_added, device=device)
    W_out_spawn = torch.zeros(n_added, d_model, device=device)
    
    for i in range(d_ff_old):
        # We'll split each parent into (growth_factor-1) children
        w_in = W_in_old[:, i]
        w_out = W_out_old[i, :]
        
        in_idx = (w_in.abs() > 1e-5).nonzero(as_tuple=True)[0]
        out_idx = (w_out.abs() > 1e-5).nonzero(as_tuple=True)[0]
        
        if len(in_idx) > 1 and len(out_idx) > 1:
            child_idx_in = in_idx[len(in_idx)//2:]
            child_idx_out = out_idx[len(out_idx)//2:]
            
            # Target child column (wrap around if needed)
            c = i % n_added
            
            W_in_spawn[child_idx_in, c] = w_in[child_idx_in]
            W_out_spawn[c, child_idx_out] = w_out[child_idx_out]
            
            # Zero out parent
            W_in_old[child_idx_in, i] = 0.0
            W_out_old[i, child_idx_out] = 0.0
            
    # Concatenate
    ffn_layer.W_in = nn.Parameter(torch.cat([W_in_old, W_in_spawn], dim=1))
    ffn_layer.b_in = nn.Parameter(torch.cat([b_in_old, b_in_spawn], dim=0))
    ffn_layer.W_out = nn.Parameter(torch.cat([W_out_old, W_out_spawn], dim=0))
    ffn_layer.d_ff = d_ff_new
    
    return d_ff_new

# ======================================================================
# TRAINING LOOP
# ======================================================================
def run_transformer_experiment(base_quant, is_ageing, args, device, train_data, val_data, l3_count=0):
    regime_name = 'Progressive Ageing' if is_ageing else 'Fixed FP32'
    print(f"\n=======================================================")
    print(f"TRANSFORMER RUN: Quant={base_quant} | FPE={regime_name} | L3Layers={l3_count}")
    print(f"=======================================================")
    
    run_name = f"transformer_l3({l3_count})_{base_quant}_{regime_name.replace(' ', '_')}"
    wandb.init(
        project="superposition-metrics",
        name=run_name,
        config={
            **vars(args),
            "base_quant": base_quant,
            "is_ageing": is_ageing,
            "l3_count": l3_count,
            "architecture": "transformer_base"
        }
    )
    
    vocab_size = 50257
    model = ProgressiveTransformer(
        vocab_size, d_model=64, n_heads=4, d_ff_start=args.d_ff_start, 
        base_quant=base_quant, is_ageing=is_ageing, l3_count=l3_count
    ).to(device)
    
    base_lr = 5e-3 if base_quant in ['Ternary', 'iq2_xxs', 'q2_k'] else 1e-3
    base_wd = 0.0 if base_quant in ['Ternary', 'iq2_xxs', 'q2_k'] else 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_wd)
    
    steps = args.n_steps
    log_interval = args.log_interval
    batch_size = args.batch_size
    seq_len = 256
    
    losses = []
    eff_ranks = []
    fpe_events = []
    proxy_flops = []
    
    best_erank = 0.0
    plateau_counter = 0
    accumulated_flops = 0.0
    
    # Parameter scaling cost heuristic
    def get_step_flops():
        # Estimate flops across all 3 layers uniformly
        return model.d_model * model.layers[0].ffn.d_ff * 2 * batch_size * seq_len * len(model.layers)
    
    for step in range(steps):
        t0 = time.time()
        xb, yb = get_batch(train_data, seq_len, batch_size, device)
        
        lr = get_lr(step, base_lr, steps)
        for pg in optimizer.param_groups:
             pg["lr"] = lr
             
        optimizer.zero_grad()
        logits, all_hidden = model(xb, return_hidden=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        loss.backward()
        
        # Enforce FPE Orthogonal Disjoint structure dynamically over all layers
        for layer in model.layers:
            sparsity_mask_in = (layer.ffn.W_in.abs() > 0).float()
            layer.ffn.W_in.grad.data *= sparsity_mask_in
            sparsity_mask_out = (layer.ffn.W_out.abs() > 0).float()
            layer.ffn.W_out.grad.data *= sparsity_mask_out
        
        optimizer.step()
        accumulated_flops += get_step_flops()
        
        if step % log_interval == 0:
            with torch.no_grad():
                model.eval()
                xv, yv = get_batch(val_data, seq_len, batch_size, device)
                val_logits, val_all_hidden = model(xv, return_hidden=True)
                val_loss = F.cross_entropy(val_logits.reshape(-1, val_logits.size(-1)), yv.reshape(-1)).item()
                # Track average Effective Rank across all transformer heights
                erank = sum(compute_effective_rank(h) for h in val_all_hidden) / len(val_all_hidden)
                model.train()
                
            losses.append(val_loss)
            eff_ranks.append(erank)
            proxy_flops.append(accumulated_flops)
            
            # TRIGGER FPE DETONATION!
            if erank > best_erank + args.tolerance:
                best_erank = erank
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            if plateau_counter >= args.patience:
                if model.layers[0].ffn.d_ff < args.d_ff_max:
                    print(f"  [Step {step}] 🎯 Triggering FPE! EffRank plateaued at {erank:.2f}")
                    for layer in model.layers:
                        new_d_ff = geometric_detonate_layer(layer.ffn, growth_factor=args.growth_factor)
                    
                    base_lr *= 0.5
                    lr = get_lr(step, base_lr, steps)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=base_wd)
                    best_erank = 0.0 # reset search
                    plateau_counter = 0
                    fpe_events.append({'step': step, 'flops': accumulated_flops, 'd_ff': new_d_ff, 'erank': erank})
                    print(f"  --> Detonated all layers to {new_d_ff} widths!")
                elif plateau_counter >= args.patience * 2:
                    print(f"  [Step {step}] 🛑 Early stopping! Model saturated at max width {args.d_ff_max}.")
                    break

            if step % (log_interval * 5) == 0:
                 print(f"  Step {step:4d} | Val Loss {val_loss:.4f} | E-Rnk {erank:4.2f} | d_ff {model.layers[0].ffn.d_ff}")

            wandb.log({
                "train/loss": loss.item(),
                "val/loss": val_loss,
                "metrics/effective_rank": erank,
                "model/d_ff": model.layers[0].ffn.d_ff,
                "train/step": step,
                "train/accumulated_flops": accumulated_flops
            })

    wandb.finish()
    return {
        'losses': losses,
        'eff_ranks': eff_ranks,
        'proxy_flops': proxy_flops,
        'fpe_events': fpe_events
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_ff_start", type=int, default=64)
    parser.add_argument("--d_ff_max", type=int, default=512)
    parser.add_argument("--growth_factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading TinyStories dataset (Subset)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Tokenize a chunk of text for quick local run
    text = " ".join([ds[i]['text'] for i in range(1000)])
    tokens = np.array(tokenizer.encode(text))
    
    split_idx = int(0.9 * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]
    
    quants = ['W8A16', 'W4A8', 'Ternary', 'q2_k', 'iq2_xxs']
    regimes = [{'name': 'Ageing', 'is_ageing': True}, {'name': 'Fixed', 'is_ageing': False}]
    l3_counts = [0, 1, 2, 3]
    
    results = {}
    for l3c in l3_counts:
        for q in quants:
            for r in regimes:
                k = f"L3_{l3c}_{q}_{r['name']}"
                results[k] = run_transformer_experiment(q, r['is_ageing'], args, device, train_data, val_data, l3_count=l3c)
            
    # Visualize FLOPs vs Loss into 4 Separate Grids
    for l3c in l3_counts:
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle(f"Progressive Quantization Matrix ({l3c} L3 Layers)", fontsize=16)
        
        for c_idx, q in enumerate(quants):
            ax_loss = axes[0, c_idx]
            ax_erank = axes[1, c_idx]
            
            ax_loss.set_title(f"{q} Base Quantization")
            ax_loss.set_ylabel("Validation Loss")
            ax_erank.set_ylabel("Effective Rank")
            ax_erank.set_xlabel("Accumulated FLOPS Proxy")
            
            for r in regimes:
                k = f"L3_{l3c}_{q}_{r['name']}"
                res = results[k]
                color = 'blue' if r['is_ageing'] else 'orange'
                
                ax_loss.plot(res['proxy_flops'], res['losses'], label=f"{r['name']}", color=color, alpha=0.8)
                ax_erank.plot(res['proxy_flops'], res['eff_ranks'], label=f"{r['name']}", color=color, alpha=0.8)
                
                for ev in res['fpe_events']:
                    ax_loss.axvline(x=ev['flops'], color=color, linestyle='--', alpha=0.3)
                    ax_erank.axvline(x=ev['flops'], color=color, linestyle='--', alpha=0.3)
    
            ax_loss.legend()
            ax_erank.legend()
    
        plt.tight_layout()
        output_path = f"../outputs/progressive_quant_transformer_l3_{l3c}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Matrix graph saved to {output_path}")

if __name__ == "__main__":
    main()

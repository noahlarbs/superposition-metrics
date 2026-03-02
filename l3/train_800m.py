import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import wandb
import tqdm
from transformers import AutoTokenizer

from l3_layer import L3, valid_collate_fn, train_lzw

# -----------------------------------------------------------------------------
# Hardware setup
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# Derived Configuration for 800M 3 L3 Layer Model
SEQ_LEN = 2048
BATCH_TOKENS = 131072
BATCH_SIZE = BATCH_TOKENS // SEQ_LEN # 64 sequences
TOTAL_TOKENS = 10_000_000_000 # 10 Billion
MAX_ITERS = TOTAL_TOKENS // BATCH_TOKENS # ~76,293 iterations
TARGET_EMBEDS = 710000 

print(f"Loading Tokenizer on {device}...")
tok = AutoTokenizer.from_pretrained('unsloth/gemma-7b') # >180K vocab requirement (256,000 vocabulary size)
VOCAB_SIZE = tok.vocab_size

class ProxTok:
    def __init__(self, t):
        self.t = t
        self.n_vocab = t.vocab_size
    def encode(self, txt):
        return self.t.encode(txt)

# -----------------------------------------------------------------------------
# Model Definition (Llama 3 Base with L3 Mix)
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

class LlamaAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class LlamaMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class HybridLlamaBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, is_l3=False, l3_kwargs_init=None):
        super().__init__()
        self.attention_norm = LlamaRMSNorm(dim)
        self.attention = LlamaAttention(dim, n_heads)
        self.ffn_norm = LlamaRMSNorm(dim)
        self.is_l3 = is_l3
        
        if is_l3:
            # Uses parameters explicitly provided out of 800M 3 L3 config frame
            self.feed_forward = L3(h=dim, n_emb=l3_kwargs_init['n_emb'], 
                                   d_emb=l3_kwargs_init['d_emb'], 
                                   d_up=l3_kwargs_init['d_up'])
        else:
            self.feed_forward = LlamaMLP(dim, hidden_dim)
            
    def forward(self, x, freqs_cis, l3_kwargs=None):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        
        # Either passes into standard MLP, or L3 layer requires external sequence mapping kwargs
        if self.is_l3:
            assert l3_kwargs is not None, "Must provide colated sequence data to L3 forward block"
            fw, bw, seq_sort, keep_cols, emb_alloc, starts, ends = l3_kwargs
            out = self.feed_forward(h, fw, bw, seq_sort, keep_cols, emb_alloc, starts, ends, bb=512)
        else:
            out = self.feed_forward(self.ffn_norm(h))
            
        return h + out

class LlamaHybridL3Model(nn.Module):
    def __init__(self, vocab_size=180000, n_layers=20, dim=1024, n_heads=32, hidden_dim=4096, 
                 l3_indices=[4, 10, 16], n_emb_total=710000, d_emb=1024, d_up=4096):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.l3_indices = l3_indices
        
        l3_init = {'n_emb': n_emb_total, 'd_emb': d_emb, 'd_up': d_up}
        
        self.layers = nn.ModuleList([
            HybridLlamaBlock(dim, n_heads, hidden_dim, is_l3=(i in l3_indices), l3_kwargs_init=l3_init)
            for i in range(n_layers)
        ])
        self.norm = LlamaRMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight # tie weights
        
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, 8192).to(device)
        
    def forward(self, input_ids, l3_kwargs=None, targets=None):
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            h = layer(h, freqs_cis, l3_kwargs)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# -----------------------------------------------------------------------------
# Data Loading and LZW setup

# -----------------------------------------------------------------------------
# Streaming Dataloader for OLMO2 Dataset
print("Initializing OLMO2 dataset streaming via HuggingFace...")
dataset = load_dataset('UW/olmo-mix-1124-subset-p99', split='train', streaming=True)
data_iter = iter(dataset)

def init_lzw():
    print("Initializing Data Loaders and LZW embedding map (Allocating 710,000 Embeddings...) ")

    # Fetch real data from the stream to train the LZW tree
    lzw_text = ""
    while len(lzw_text) < 1_000_000: # Need enough data to exhaust TARGET_EMBEDS tokens
        try:
            doc = next(data_iter)
            lzw_text += doc['text'] + "\n"
        except StopIteration:
            break
            
    with open("/tmp/lzw_train.txt", "w") as f:
        f.write(lzw_text)

    alloc = train_lzw(["/tmp/lzw_train.txt"], ProxTok(tok), k=512, target=TARGET_EMBEDS)
    orig_n_embs = sum(alloc)

    emb_alloc_list = []
    for tid, count in enumerate(alloc):
        emb_alloc_list.extend([tid] * count)
        
    emb_alloc_tensor = torch.tensor(emb_alloc_list, dtype=torch.long)
    bounds = torch.cumsum(torch.cat((torch.tensor([0]), torch.tensor(alloc, dtype=torch.long))), dim=0)
    
    return orig_n_embs, emb_alloc_tensor, bounds

orig_n_embs, emb_alloc_tensor, bounds = init_lzw()

def fetch_data():
    """Lazily fetches BATCH_SIZE * SEQ_LEN tokens from the raw OLMO2 dataset."""
    tokens = []
    while len(tokens) < BATCH_SIZE * SEQ_LEN + 1:
        try:
            doc = next(data_iter)
            tokens.extend(tok.encode(doc['text']))
        except StopIteration:
            break
            
    # Fallback to random if dataset exhausted (or handle re-init)
    if len(tokens) < BATCH_SIZE * SEQ_LEN + 1:
        print("Dataset exhausted, falling back to dummy data for this batch.")
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        return x, y
        
    # Tensorize and shape into batches
    tokens_tensor = torch.tensor(tokens[:BATCH_SIZE * SEQ_LEN + 1], dtype=torch.long)
    x = tokens_tensor[:-1].view(BATCH_SIZE, SEQ_LEN)
    y = tokens_tensor[1:].view(BATCH_SIZE, SEQ_LEN)
    
    return x, y

def main():
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = LlamaHybridL3Model(
        vocab_size=VOCAB_SIZE, 
        n_layers=20, 
        dim=1024, 
        n_heads=32, 
        hidden_dim=4096, 
        l3_indices=[4, 10, 16], 
        n_emb_total=orig_n_embs, 
        d_emb=1024, 
        d_up=4096
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=3e-5)

    wandb.init(project="l3-800m-test", config={
        "decoder_layers": 20,
        "hidden_dim": 1024,
        "context_len": 2048,
        "batch_size": 131072,
        "peak_lr": 3e-4,
        "min_lr": 3e-5,
        "weight_decay": 0.1,
        "l3_embeds": orig_n_embs,
        "l3_layers": [4, 10, 16]
    })

    print(f"Estimated Training Time on 1x A100: ~88 hours (3.7 days) to process 10B tokens.")
    model.train()

    print("Starting Training Loop...")
    for it in range(MAX_ITERS):
        t0 = time.time()
        
        X, Y = fetch_data()
        
        batch_for_collate = [(X[i], emb_alloc_tensor, bounds) for i in range(BATCH_SIZE)]
        _, fw, bw, seq_sort, keep_cols, emb_alloc_collate, starts, ends = valid_collate_fn(batch_for_collate)
        
        X, Y = X.to(device), Y.to(device)
        l3_kwargs = (fw.to(device), bw.to(device), seq_sort.to(device), 
                     keep_cols.to(device), emb_alloc_collate.to(device), 
                     starts.to(device), ends.to(device))
                     
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16):
            logits, loss = model(X, targets=Y, l3_kwargs=l3_kwargs)
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        t1 = time.time()
        dt = t1 - t0
        toks_sec = BATCH_TOKENS / dt
        
        if it % 10 == 0:
            print(f"step {it} | loss: {loss.item():.4f} | dt: {dt:.2f}s | tok/s: {toks_sec:.2f}")
            wandb.log({"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0], "tok_sec": toks_sec})

if __name__ == "__main__":
    main()

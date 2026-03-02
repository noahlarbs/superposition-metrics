import torch
import torch.nn as nn
import tqdm

# Optional flex_attention import required for the alternative forward pass
try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    pass

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def train_lzw(files, tok, k, target):
    lzw_counter = {}
    for s in range(tok.n_vocab):
        lzw_counter[(s,)] = 0
    for fn in files:
        f = open(fn).readlines()
        for line in tqdm.tqdm(f, mininterval=1):
            toks = tok.encode(line)
            last = 0
            cur = 1
            while cur < len(toks):
                while cur < len(toks) and tuple(toks[last:cur]) in lzw_counter:
                    cur += 1
                if cur > last+1:
                    lzw_counter[tuple(toks[last:cur-1])] += 1
                lzw_counter[tuple(toks[last:cur])] = 1
                last = cur
                cur += 1
    lzw_counter = sorted(list(lzw_counter.items()), key=lambda x: x[1], reverse=True)
    alloc = [1 for _ in range(max(tok.n_vocab, target))]
    n_alloc = tok.n_vocab
    i = 0
    while n_alloc < target and i < len(lzw_counter):
        token_id = lzw_counter[i][0][-1]
        if token_id < len(alloc):
            if alloc[token_id] < k:
                alloc[token_id] += 1
                n_alloc += 1
        i += 1
        
    # If we run out of tokens in the corpus to reach `target`, just pad the first token (ID 0).
    while n_alloc < target:
        if alloc[0] < k:
            alloc[0] += 1
            n_alloc += 1
        else:
            break # Or pick the next token, but padding is fine for this demo limit
            
    return alloc[:max(target, tok.n_vocab)] # Ensure precise match without OOB

def valid_collate_fn(batch):
    # Data collation function to produce inputs needed for L3 forward below
    # seqs are tokenized inputs
    # emb_alloc is the token ID for that embedding
    # bounds[i] is the start the region in emb_alloc corresponding to token i
    seqs = torch.stack([_[0] for _ in batch], dim=0)
    emb_alloc = batch[0][1]
    bounds = batch[0][2]
    seq_sort, fw = torch.sort(seqs.flatten())
    bw = torch.zeros(len(seq_sort), dtype=torch.int64)
    bw[fw] = torch.arange(len(seq_sort))
    unique, cts = torch.unique(seq_sort, return_counts=True)
    keep_cols = []
    starts = []
    ends = []
    for i in range(len(unique)):
        tidx = unique[i]
        tct = cts[i]
        starts += [len(keep_cols)] * tct
        keep_cols += list(range(bounds[tidx], bounds[tidx+1]))
        ends += [len(keep_cols)] * tct
    keep_cols = torch.tensor(keep_cols)
    starts = torch.tensor(starts)
    ends = torch.tensor(ends)
    return seqs, fw, bw, seq_sort, keep_cols, emb_alloc, starts, ends

class L3(torch.nn.Module):
    def __init__(self, h, n_emb, d_emb, d_up):
        super().__init__()
        self.d_emb = d_emb
        self.w_k = nn.Linear(h, n_emb, bias=False)
        self.w_v = nn.Linear(d_emb, n_emb, bias=False)
        self.w_mix = nn.Linear(d_up + h, h, bias=False)
        self.w_up = nn.Linear(d_emb, d_up, bias=False)
        self.norm_in = LlamaRMSNorm(h)
        self.norm_out = LlamaRMSNorm(d_up)

    @torch.compile(mode='max-autotune', fullgraph=True)
    def mask_logits_gemm_(self, A, B, C, seq_sort, last_token):
        score = A @ B.T
        score = torch.where(seq_sort.view(-1, 1) == last_token.view(1, -1), score, -float(
            'inf'))
        return score.softmax(dim=-1) @ C

    def mask_logits_gemm(self, A, B, C, seq_sort, last_token, starts, ends):
        b, t, d = A.shape
        out = torch.zeros(b, t, C.shape[-1], device=A.device, dtype=A.dtype)
        for i in range(b):
            start = starts[i]
            end = ends[i]
            out[i] = self.mask_logits_gemm_(A[i], B[start:end], C[start:end], seq_sort[i],
                                            last_token[start:end])
        return out

    def forward(self, input, fw, bw, seq_sort, keep_cols, emb_alloc, starts, ends, bb=512):
        # bb is the size of the vertical dim of the block diagonal
        b, t, d = input.shape
        A = self.norm_in(input)
        B = self.w_k.weight
        C = self.w_v.weight
        A = A.reshape(-1, d)[fw].reshape(-1, bb, d)
        B = B[keep_cols]
        C = C[keep_cols]
        emb_alloc = emb_alloc[keep_cols]
        seq_sort = seq_sort.reshape(-1, bb)
        starts = starts.reshape(-1, bb).min(dim=-1).values
        ends = ends.reshape(-1, bb).max(dim=-1).values
        comb_embs = self.mask_logits_gemm(A, B, C, seq_sort, emb_alloc, starts, ends)
        comb_embs = comb_embs.reshape(-1, self.d_emb)[bw].reshape(b, t, self.d_emb)
        comb_embs = self.w_up(comb_embs)
        return self.w_mix(torch.concat([self.norm_out(comb_embs), input], dim=-1))

    def forward_flexattn(self, input, fw, bw, seq_sort, keep_cols, emb_alloc):
        b, t, d = input.shape
        emb_alloc = emb_alloc[keep_cols]
        
        def mask_fn(b, h, q_idx, kv_idx):
            return seq_sort[b, q_idx] == emb_alloc[kv_idx]
            
        mask = flex_attention.create_block_mask(
            mask_fn, B=b, H=None, Q_LEN=t, KV_LEN=emb_alloc.shape[0], _compile=True)
            
        A = self.norm_in(input).reshape(-1, d)[fw].reshape(b, t, d).unsqueeze(1)
        B = self.w_k.weight.unsqueeze(0).unsqueeze(1)
        C = self.w_v.weight.unsqueeze(0).unsqueeze(1)
        
        comb_embs = torch.compile(flex_attention.flex_attention, mode='max-autotune')(
            A, B, C, block_mask=mask, enable_gqa=True)
            
        comb_embs = comb_embs.reshape(-1, self.d_emb)[bw].reshape(b, t, self.d_emb)
        comb_embs = self.w_up(comb_embs)
        return self.w_mix(torch.concat([self.norm_out(comb_embs), input], dim=-1))

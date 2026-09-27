"""The attention-scored KV selection and its compact buffer."""

import math
import time
import types

import torch
from torch.nn import functional as F

from bytecross.harness import apply_rotary_emb, install, project_out, project_qkv, splitk_decode


def scoring_forward(store, layer, prompt_len, keep, sinks, window, pool):
    # Dense attention of the observation window; also records the top `keep` positions of each KV head.
    def forward(self, x, freqs_cis, mask, input_pos=None):
        b, s, _ = x.shape
        kv = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv, kv], dim=-1)
        q = apply_rotary_emb(q.view(b, s, self.n_head, self.head_dim), freqs_cis).transpose(1, 2)
        k = apply_rotary_emb(k.view(b, s, self.n_local_heads, self.head_dim), freqs_cis).transpose(1, 2)
        v = v.view(b, s, self.n_local_heads, self.head_dim).transpose(1, 2)
        k_all, v_all = self.kv_cache.update(input_pos, k, v)
        groups = self.n_head // self.n_local_heads
        k_all = k_all.repeat_interleave(groups, dim=1)
        v_all = v_all.repeat_interleave(groups, dim=1)

        scores = torch.matmul(q, k_all[:, :, :prompt_len].transpose(-1, -2)) / math.sqrt(self.head_dim)
        future = torch.arange(prompt_len, device=x.device) > input_pos.long().view(-1, 1)
        probs = torch.softmax(scores.masked_fill(future, float("-inf")).float(), dim=-1).sum(dim=2)
        probs = probs.view(b, self.n_local_heads, groups, prompt_len).sum(dim=2)  # pool the query heads of a group
        if pool > 1:
            probs = F.avg_pool1d(probs, kernel_size=pool, stride=1, padding=pool // 2)[..., :prompt_len]
        forced = torch.zeros(prompt_len, dtype=torch.bool, device=x.device)
        forced[:sinks] = True
        forced[prompt_len - window:] = True
        probs = probs.masked_fill(forced, float("inf"))
        store[layer] = probs.topk(keep, dim=-1).indices[0].sort(dim=-1).values

        y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        return self.wo(y.transpose(1, 2).reshape(b, s, self.dim))

    return forward


def compact_forward(split_k):
    # Decode over the compact buffer: `c_static` selected entries, then the entries of later steps.
    # The full cache is still updated, so a step can switch back to it.
    def forward(self, x, freqs_cis, mask, input_pos=None):
        b, s, _ = x.shape
        q, k, v = project_qkv(self, x, freqs_cis)
        self.kv_cache.update(input_pos, k, v)
        slot = input_pos.to(torch.long) - self.c_base + self.c_static
        self.ck[:, :, slot] = k
        self.cv[:, :, slot] = v
        if split_k:
            seqlens = (slot[-1:] + 1).to(torch.int32)
            y = splitk_decode(q.transpose(1, 2).contiguous(), self.ck.transpose(1, 2), self.cv.transpose(1, 2), seqlens)
        else:
            n = int(slot[-1]) + 1
            groups = self.n_head // self.n_local_heads
            y = F.scaled_dot_product_attention(q, self.ck[:, :, :n].repeat_interleave(groups, dim=1),
                                               self.cv[:, :, :n].repeat_interleave(groups, dim=1))
            y = y.transpose(1, 2)
        return project_out(self, y.reshape(b, s, self.dim))

    return forward


@torch.no_grad()
def gather(model, idx, static, extra, base):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, layer in enumerate(model.layers):
        a = layer.attention
        kc, vc = a.kv_cache.k_cache, a.kv_cache.v_cache
        shape = (kc.shape[0], kc.shape[1], static + extra + 8, kc.shape[3])
        a.register_buffer("ck", torch.zeros(shape, dtype=kc.dtype, device=kc.device), persistent=False)
        a.register_buffer("cv", torch.zeros(shape, dtype=kc.dtype, device=kc.device), persistent=False)
        g = idx[i][None, :, :, None].expand(1, -1, -1, kc.shape[3])
        a.ck[:, :, :static] = torch.gather(kc, 2, g)
        a.cv[:, :, :static] = torch.gather(vc, 2, g)
        a.c_static, a.c_base = static, base
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def select_and_compact(model, tokens, first_pos, static, extra, sinks=4, window=64, pool=7):
    """Score cache positions [0, first_pos) with the last `window` of them and keep `static` per KV head.

    Expects a dense prefill of those positions. Returns the seconds of the scoring pass and the gather."""
    saved = [l.feed_forward.forward for l in model.layers]
    store = {}
    for i, l in enumerate(model.layers):
        l.attention.forward = types.MethodType(
            scoring_forward(store, i, first_pos, static, sinks, window, pool), l.attention)
        l.feed_forward.forward = types.MethodType(type(l.feed_forward).forward, l.feed_forward)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pos = torch.arange(first_pos - window, first_pos, device=tokens.device, dtype=torch.int)
    model(tokens[first_pos - window:first_pos].view(1, -1), pos)
    torch.cuda.synchronize()
    score_s = time.perf_counter() - t0
    for l, f in zip(model.layers, saved):
        l.feed_forward.forward = f
    gather_s = gather(model, store, static, extra, first_pos)
    install(model, compact_forward(split_k=False))
    return score_s, gather_s


def random_compact(model, context, static, extra, seed=0):
    # Timing only: the step reads `static` gathered entries whichever positions they come from.
    gen = torch.Generator().manual_seed(seed + context)
    heads = model.layers[0].attention.n_local_heads
    device = model.layers[0].attention.kv_cache.k_cache.device
    idx = [torch.stack([torch.randperm(context, generator=gen)[:static].sort().values for _ in range(heads)]).to(device)
           for _ in model.layers]
    secs = gather(model, idx, static, extra, context)
    install(model, compact_forward(split_k=True))
    return secs

"""Model loading, TEAL projection sparsity, split-K decode attention and dense prefill."""

import math
import os
import sys
import time
import types
from pathlib import Path

import torch

TEAL_DIR = Path(os.environ.get("TEAL_DIR", "TEAL")).resolve()
if not (TEAL_DIR / "gpt-fast/model.py").exists():
    raise ImportError(f"TEAL not found at {TEAL_DIR}. Clone https://github.com/FasterDecoding/TEAL and set TEAL_DIR")
sys.path[:0] = [str(TEAL_DIR / "gpt-fast"), str(TEAL_DIR)]

import model as gpt_fast  # noqa: E402
from model import KVCache, apply_rotary_emb, find_multiple  # noqa: E402

DEFAULT_HIST = TEAL_DIR / "models/Llama-3-8B/histograms"

gpt_fast.transformer_configs.update({
    "Llama-3.1-8B-Instruct": dict(block_size=35000, n_layer=32, n_head=32, n_local_heads=8, dim=4096,
                                  intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "Llama-3.2-3B-Instruct": dict(block_size=131072, n_layer=28, n_head=24, n_local_heads=8, dim=3072,
                                  intermediate_size=8192, vocab_size=128256, rope_base=500000),
})
ROPE_SCALING = {
    "llama-3.1": dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    "llama-3.2": dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
}


def rope_cache(seq_len, head_dim, base, dtype, scaling=None):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    if scaling is not None:
        # llama3 frequency scaling (transformers' _compute_llama3_parameters)
        factor, lo, hi = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]
        old = scaling["original_max_position_embeddings"]
        wavelen = 2 * math.pi / freqs
        smooth = (old / wavelen - lo) / (hi - lo)
        mid = (1 - smooth) * freqs / factor + smooth * freqs
        freqs = torch.where(wavelen < old / hi, freqs, torch.where(wavelen > old / lo, freqs / factor, mid))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)
    rot = torch.polar(torch.ones_like(angles), angles)
    return torch.stack([rot.real, rot.imag], dim=-1).to(dtype)


class LazyCausalMask:
    # gpt-fast builds an L x L mask; beyond ~64K tokens it does not fit, so build only the rows asked for
    def __init__(self, length, device):
        self.cols = torch.arange(length, device=device)

    def __getitem__(self, idx):
        pos = idx[-1].view(-1)
        return (self.cols[None, :] <= pos[:, None])[None, None]


def setup_caches(self, max_batch_size, max_seq_length):
    head_dim = self.config.dim // self.config.n_head
    max_seq_length = find_multiple(max_seq_length, 8)
    self.max_seq_length, self.max_batch_size = max_seq_length, max_batch_size
    dtype = self.output.weight.dtype
    for b in self.layers:
        b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)
    self.freqs_cis = rope_cache(self.config.block_size, head_dim, self.config.rope_base, dtype,
                                getattr(self.config, "rope_scaling", None))
    if getattr(self, "lazy_mask", False):
        self.causal_mask = LazyCausalMask(max_seq_length, self.output.weight.device)
    else:
        self.causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool))


gpt_fast.Transformer.setup_caches = setup_caches


def load_model(checkpoint, max_len, device="cuda", dtype=torch.float16):
    name = Path(checkpoint).parent.name
    with torch.device("meta"):
        model = gpt_fast.Transformer.from_name(name)
    state = torch.load(str(checkpoint), mmap=True, weights_only=True)
    model.load_state_dict(state.get("model", state), assign=True)
    model = model.to(device=device, dtype=dtype)
    model.config.block_size = max(model.config.block_size, max_len)
    model.config.rope_scaling = next((v for k, v in ROPE_SCALING.items() if k in name.lower()), None)
    return model.eval()


def sparsify_projections(model, hist, keep):
    """TEAL thresholds for all seven projections at the given keep ratio, with TEAL's sparse GEMV."""
    from distribution import Distribution
    from kernels.sparse_gemv import SparseGEMV, SparseQKVGEMV
    q = 0.5 + 0.5 * round(1 - keep, 6)
    hist = str(hist or DEFAULT_HIST)
    dev = model.layers[0].feed_forward.w1.weight.device.type
    for i, layer in enumerate(model.layers):
        def threshold(family, hidden):
            return Distribution(os.path.join(hist, f"layer-{i}", family), hidden).icdf(q).item()

        ff, at = layer.feed_forward, layer.attention
        ff.gemv1_kernel = SparseGEMV.initialize("sparse_gemv", dev)
        ff.gemv1 = ff.gemv1_kernel.operator(True)
        ff.gemv2_kernel = SparseGEMV.initialize("sparse_gemv", dev)
        ff.gemv2 = ff.gemv2_kernel.operator(True)
        ff.thresh_gate = ff.thresh_up = threshold("mlp", "h1")
        ff.thresh_down = threshold("mlp", "h2")
        ff.sparsity_bin = 0

        at.gemv1_kernel = SparseQKVGEMV.initialize("sparse_qkv_gemv", dev)
        at.gemv1 = at.gemv1_kernel.operator(True)
        at.gemv2_kernel = SparseGEMV.initialize("sparse_gemv", dev)
        at.gemv2 = at.gemv2_kernel.operator(True)
        at.thresh_q = at.thresh_k = at.thresh_v = threshold("self_attn", "h1")
        at.thresh_o = threshold("self_attn", "h2")
        at.sparsity_bin = 0

        # the sparse kernels read column-major weights
        for w in (ff.w1, ff.w3, ff.w2, at.wqkv, at.wo):
            w.weight.data = w.weight.data.T.contiguous().T
        ff.apply_monkeypatch()
        at.apply_monkeypatch()
        torch.cuda.empty_cache()


def project_qkv(attn, x, freqs_cis):
    b, s, _ = x.shape
    kv = attn.n_local_heads * attn.head_dim
    if hasattr(attn, "gemv1"):
        qkv = attn.gemv1(x, attn.wqkv.weight, attn.thresh_q, attn.thresh_k, attn.thresh_v, attn.sparsity_bin, kv)
    else:
        qkv = attn.wqkv(x)
    q, k, v = qkv.split([attn.dim, kv, kv], dim=-1)
    q = apply_rotary_emb(q.view(b, s, attn.n_head, attn.head_dim), freqs_cis)
    k = apply_rotary_emb(k.view(b, s, attn.n_local_heads, attn.head_dim), freqs_cis)
    v = v.view(b, s, attn.n_local_heads, attn.head_dim)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def project_out(attn, y):
    if hasattr(attn, "gemv2"):
        return attn.gemv2(y, attn.wo.weight, attn.thresh_o, attn.sparsity_bin)
    return attn.wo(y)


# flash_attn_with_kvcache (split-K) as an opaque op, so the compiled decode step has no graph break
@torch.library.custom_op("bytecross::splitk_decode", mutates_args=(),
                         schema="(Tensor q, Tensor k_cache, Tensor v_cache, Tensor seqlens) -> Tensor")
def splitk_decode(q, k_cache, v_cache, seqlens):
    from flash_attn import flash_attn_with_kvcache
    return flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=seqlens, causal=False).contiguous()


@splitk_decode.register_fake
def _(q, k_cache, v_cache, seqlens):
    return torch.empty_like(q)


def splitk_forward(self, x, freqs_cis, mask, input_pos=None):
    b, s, _ = x.shape
    q, k, v = project_qkv(self, x, freqs_cis)
    self.kv_cache.update(input_pos, k, v)
    seqlens = (input_pos[-1:] + 1).to(torch.int32)
    y = splitk_decode(q.transpose(1, 2).contiguous(), self.kv_cache.k_cache.transpose(1, 2),
                      self.kv_cache.v_cache.transpose(1, 2), seqlens)
    return project_out(self, y.reshape(b, s, self.dim))


def install(model, forward):
    for layer in model.layers:
        layer.attention.forward = types.MethodType(forward, layer.attention)


def _flash_prefill(self, x, freqs_cis, mask, input_pos=None):
    from flash_attn import flash_attn_func
    b, s, _ = x.shape
    kv = self.n_local_heads * self.head_dim
    q, k, v = self.wqkv(x).split([self.dim, kv, kv], dim=-1)
    q = apply_rotary_emb(q.view(b, s, self.n_head, self.head_dim), freqs_cis)
    k = apply_rotary_emb(k.view(b, s, self.n_local_heads, self.head_dim), freqs_cis)
    v = v.view(b, s, self.n_local_heads, self.head_dim)
    kc, vc = self.kv_cache.update(input_pos, k.transpose(1, 2), v.transpose(1, 2))
    end = int(input_pos[-1]) + 1
    # bottom-right causal alignment: each query of the chunk sees the cache up to its own position
    y = flash_attn_func(q, kc[:, :, :end].transpose(1, 2), vc[:, :, :end].transpose(1, 2), causal=True)
    return self.wo(y.reshape(b, s, self.dim))


@torch.no_grad()
def prefill(model, tokens, chunk=512, flash=True):
    """Dense prefill of tokens[0:n] at positions 0..n-1, whatever decode forwards are installed."""
    saved = [(l.attention.forward, l.feed_forward.forward) for l in model.layers]
    for l in model.layers:
        attn = _flash_prefill if flash else type(l.attention).forward
        l.attention.forward = types.MethodType(attn, l.attention)
        l.feed_forward.forward = types.MethodType(type(l.feed_forward).forward, l.feed_forward)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    finite = True
    try:
        for start in range(0, tokens.numel(), chunk):
            end = min(start + chunk, tokens.numel())
            pos = torch.arange(start, end, device=tokens.device, dtype=torch.int)
            finite &= bool(torch.isfinite(model(tokens[start:end].view(1, -1), pos)).all())
    finally:
        for l, (a, f) in zip(model.layers, saved):
            l.attention.forward, l.feed_forward.forward = a, f
    if not finite:
        raise RuntimeError(f"non-finite logits in the prefill of {tokens.numel()} tokens")
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def tokenizer(checkpoint):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(Path(checkpoint).parent)


def wikitext_ids(tok):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]


def load_for_eval(checkpoint, hist, rp, max_seq, lazy_mask=False):
    model = load_model(checkpoint, max_seq + 16)
    if rp < 1:
        sparsify_projections(model, hist, rp)
    model.lazy_mask = lazy_mask
    with torch.device("cuda"):
        model.setup_caches(1, max_seq)
    return model


def parse_setting(s):
    rp, rkv = (float(x) for x in s.split("/"))
    return rp, rkv

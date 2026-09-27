"""Byte account of one decode step (Section 3): traffic, crossover and speedup bounds."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

K = 1024


@dataclass(frozen=True)
class Model:
    layers: int
    d: int
    d_ff: int
    vocab: int
    kv_heads: int = 8
    head_dim: int = 128
    window: int = 0

    @classmethod
    def from_config(cls, cfg):
        """Dimensions from a Hugging Face config.json, given as a dict or a path."""
        if not isinstance(cfg, dict):
            cfg = json.loads(Path(cfg).read_text())
        cfg = cfg.get("text_config", cfg)
        heads = cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
        if heads * head_dim != cfg["hidden_size"]:
            # Eq. 3 counts the query and output projections as d x d
            raise ValueError("the byte account assumes num_attention_heads * head_dim == hidden_size")
        return cls(cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"],
                   kv_heads=cfg.get("num_key_value_heads", heads), head_dim=head_dim,
                   window=cfg.get("max_position_embeddings", 0))


MODELS = {
    "qwen3-8b": Model(36, 4096, 12288, 151936, window=40960),
    "llama-3.1-8b": Model(32, 4096, 14336, 128256, window=131072),
    "mistral-7b": Model(32, 4096, 14336, 32768, window=32768),
    "llama-3.1-70b": Model(80, 8192, 28672, 128256, window=131072),
    "llama-3.2-3b": Model(28, 3072, 8192, 128256, window=131072),
}


def model_for(name):
    for key, m in MODELS.items():
        if key in name.lower():
            return m
    raise KeyError(f"no dimensions for {name}")


# Bytes read by one decoder layer in 16-bit precision (Eqs. 2-4).
def mlp_bytes(m):
    return 3 * m.d * m.d_ff * 2


def attn_bytes(m):
    return (2 * m.d ** 2 + 2 * m.d * m.kv_heads * m.head_dim) * 2


def kv_bytes(m, n):
    return 2 * m.kv_heads * m.head_dim * n * 2


def other_bytes(m):
    # output head, one embedding row, and every RMSNorm weight
    return (m.vocab * m.d + m.d + (2 * m.layers + 1) * m.d) * 2


def step_bytes(m, n):
    return m.layers * (mlp_bytes(m) + attn_bytes(m) + kv_bytes(m, n)) + other_bytes(m)


def proj_params(m, ff_only=False):
    p = 3 * m.d * m.d_ff
    return p if ff_only else p + 2 * m.d ** 2 + 2 * m.d * m.kv_heads * m.head_dim


def crossover(m, rp, rkv, ff_only=False, weight_bytes=2, kv_elem_bytes=2):
    n = proj_params(m, ff_only) * (1 - rp) / (2 * m.kv_heads * m.head_dim * (1 - rkv))
    return n * weight_bytes / kv_elem_bytes


def crossover_fixed_budget(m, rp, budget):
    # r_KV = budget / n; valid when the crossing lies above the budget
    return budget + proj_params(m) * (1 - rp) / (2 * m.kv_heads * m.head_dim)


def crossover_batch(m, union_keep, rkv, batch):
    return proj_params(m) * (1 - union_keep) / (2 * batch * m.kv_heads * m.head_dim * (1 - rkv))


def bound(m, n, rp=1.0, rkv=1.0):
    total = step_bytes(m, n)
    saved = m.layers * ((mlp_bytes(m) + attn_bytes(m)) * (1 - rp) + kv_bytes(m, n) * (1 - rkv))
    return total / (total - saved)


def admissible_keep(curve, budget):
    # curve: keep ratio -> PPL increase. Lowest keep whose linearly interpolated increase stays
    # within the budget; None if even the mildest tested keep exceeds it.
    pts = sorted(curve.items(), reverse=True)
    if pts[0][1] > budget:
        return None
    for (k0, d0), (k1, d1) in zip(pts, pts[1:]):
        if d1 > budget:
            return k0 - (k0 - k1) * (budget - d0) / (d1 - d0)
    return pts[-1][0]


def matched_crossover(m, proj_curve, kv_curve, budget):
    rp, rkv = admissible_keep(proj_curve, budget), admissible_keep(kv_curve, budget)
    if rp is None or rkv is None:
        return None
    return crossover(m, rp, rkv)


def print_table1(rp=0.5, rkv=0.3):
    print(f"Byte crossover at r_P={rp}, r_KV={rkv}")
    print(f"{'model':14s} {'d':>5s} {'d_ff':>6s} {'n*':>8s} {'n*_FF':>8s} {'window':>7s}")
    for name in ("qwen3-8b", "llama-3.1-8b", "mistral-7b", "llama-3.1-70b"):
        m = MODELS[name]
        print(f"{name:14s} {m.d:5d} {m.d_ff:6d} {crossover(m, rp, rkv) / K:7.1f}K "
              f"{crossover(m, rp, rkv, ff_only=True) / K:7.1f}K {m.window // K:6d}K")


def print_step_shares(name="qwen3-8b", ff_keep=0.5, rkv=0.3):
    m = MODELS[name]
    print(f"\n{name} decode step: share of reads (%) and MB removed by FF-only {ff_keep} / KV {rkv}")
    print(f"{'context':>8s} {'MLP':>6s} {'Attn':>6s} {'KV':>6s} {'Other':>6s} {'MLP MB':>7s} {'KV MB':>7s}")
    for n in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        total = step_bytes(m, n)
        mlp, attn, kv = (m.layers * b for b in (mlp_bytes(m), attn_bytes(m), kv_bytes(m, n)))
        shares = [100 * b / total for b in (mlp, attn, kv, other_bytes(m))]
        print(f"{n:8d} " + " ".join(f"{s:6.1f}" for s in shares)
              + f" {mlp * (1 - ff_keep) / 1e6:7.0f} {kv * (1 - rkv) / 1e6:7.0f}")


def load_quality_curves(paths, context=32768):
    # PPL increase over dense for each keep ratio of each branch alone, from ppl.py outputs
    rows = [r for p in paths for r in json.loads(Path(p).read_text())["results"] if r["context"] == context]
    dense = next(r["ppl"] for r in rows if r["projection_keep"] == 1 and r["kv_keep"] == 1)
    proj = {r["projection_keep"]: r["ppl"] - dense for r in rows if r["kv_keep"] == 1 and r["projection_keep"] < 1}
    sel = {r["kv_keep"]: r["ppl"] - dense for r in rows if r["projection_keep"] == 1 and r["kv_keep"] < 1}
    return proj, sel


def print_matched(paths, m, budgets, context=32768):
    proj, sel = load_quality_curves(paths, context)
    print(f"{'budget':>6s} {'r_P':>6s} {'r_KV':>6s} {'n*':>8s}")
    for b in budgets:
        rp, rkv, n = admissible_keep(proj, b), admissible_keep(sel, b), matched_crossover(m, proj, sel, b)
        if n is None:
            print(f"{b:6.2f} no admissible keep ratio")
        else:
            print(f"{b:6.2f} {rp:6.3f} {rkv:6.3f} {n / K:7.1f}K")


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog,
                                 description="Byte account: Table 1, Table 9, and the byte bounds of Figure 3.")
    ap.add_argument("--model", default="llama-3.1-8b", choices=tuple(MODELS))
    ap.add_argument("--rp", type=float, default=0.5)
    ap.add_argument("--rkv", type=float, default=0.3)
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072])
    ap.add_argument("--matched", type=Path, nargs="+", default=None,
                    help="ppl.py outputs of the 32K keep-ratio sweep: quality-matched crossover (Figure 6)")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    a = ap.parse_args(argv)
    m = MODELS[a.model]
    if a.matched:
        print_matched(a.matched, m, a.budgets)
        return
    print_table1()
    print_step_shares()
    print(f"\n{a.model} byte bounds, r_P={a.rp}, r_KV={a.rkv}, n*={crossover(m, a.rp, a.rkv) / K:.1f}K")
    print(f"{'context':>8s} {'proj':>6s} {'KV':>6s} {'both':>6s}")
    for n in a.contexts:
        print(f"{n // K:7d}K {bound(m, n, a.rp):6.3f} {bound(m, n, rkv=a.rkv):6.3f} {bound(m, n, a.rp, a.rkv):6.3f}")


if __name__ == "__main__":
    main()

"""Measured latency crossings of the two branches, with intervals over all block resamples."""

import argparse
from pathlib import Path

import numpy as np

from bytecross.campaign import block_interval, load_runs, resamples
from bytecross.traffic import K, crossover, model_for


def crossing(contexts, diffs):
    # first change from projection ahead (diff < 0) to selection ahead, linearly interpolated
    for c0, c1, d0, d1 in zip(contexts, contexts[1:], diffs, diffs[1:]):
        if d0 < 0 <= d1:
            return c0 + (c1 - c0) * -d0 / (d1 - d0)
    return None


def crossing_interval(contexts, diffs, q=(0.025, 0.975)):
    # diffs: [blocks, contexts] of projection minus selection step time, paired inside each block
    diffs = np.asarray(diffs, float)
    means = diffs[resamples(len(diffs))].mean(axis=1)
    pts = [p for p in (crossing(contexts, row) for row in means) if p is not None]
    return (np.quantile(pts, q) if pts else None), len(pts), len(means)


def analyze_group(rows, group, show=True):
    g = [r for r in rows if r["group"] == group]
    proj = next(r for r in g if r["mode"] == "proj")
    sel = next(r for r in g if r["mode"] == "sel")
    ms = {(r["block"], r["context"], r["mode"]): r["ms_per_token"] for r in g}
    ctxs = sorted({r["context"] for r in g})
    blocks = sorted(b for b in {r["block"] for r in g}
                    if all((b, c, m) in ms for c in ctxs for m in ("proj", "sel")))
    P = np.array([[ms[b, c, "proj"] for c in ctxs] for b in blocks])
    S = np.array([[ms[b, c, "sel"] for c in ctxs] for b in blocks])
    D = P - S
    m = model_for(proj["model"])
    rp, rkv = proj["projection_keep"], sel["kv_keep"]
    x = crossing(ctxs, D.mean(axis=0))
    ci, hits, total = crossing_interval(ctxs, D)
    if not show:
        return (x, ci) if x is not None else None
    print(f"\n{group}: {proj['model']} on {proj['gpu']}, keep {rp} / {rkv}, "
          f"byte crossover {crossover(m, rp, rkv) / K:.1f}K, {len(blocks)} complete blocks")
    print(f"{'context':>8s} {'proj ms':>8s} {'sel ms':>8s} {'diff':>8s} {'95% interval':>18s} {'sel faster':>10s}")
    for j, c in enumerate(ctxs):
        lo, hi = block_interval(D[:, j])
        print(f"{c // K:7d}K {P[:, j].mean():8.2f} {S[:, j].mean():8.2f} {D[:, j].mean():8.3f} "
              f"[{lo:7.3f}, {hi:7.3f}] {int((D[:, j] > 0).sum()):>7d}/{len(blocks)}")
    if x is None:
        print("no crossing inside the grid")
        return None
    ci_text = f"[{ci[0] / K:.1f}K, {ci[1] / K:.1f}K]" if ci is not None else "[-]"
    print(f"crossing {x / K:.1f}K {ci_text}, {hits} of {total} block resamples cross")
    return x, ci


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description="Locate each projection/selection crossing of a campaign.")
    ap.add_argument("runs", type=Path, nargs="+", help="campaign folders; launches of one grid are pooled")
    a = ap.parse_args(argv)
    rows = load_runs(a.runs)
    for group in sorted({r["group"] for r in rows}):
        analyze_group(rows, group)


if __name__ == "__main__":
    main()

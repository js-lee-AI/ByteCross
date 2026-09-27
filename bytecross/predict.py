"""Kernel costs beyond the byte time (Table 10) and cost-aware crossing predictions (Table 2)."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from bytecross.campaign import load_runs
from bytecross.traffic import K, bound, crossover, kv_bytes, model_for, step_bytes


def branch(r):
    return r["mode"], r["projection_keep"], r["kv_keep"]


def kernel_costs(rows, m):
    # step time beyond the byte time: t_b - t_D * B_b / B_D, with t_D from the same block and context
    dense = {(r["block"], r["group"], r["context"]): r["ms_per_token"] for r in rows if r["mode"] == "dense"}
    costs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["block"], r["group"], r["context"])
        if r["mode"] != "dense" and key in dense:
            ideal = dense[key] / bound(m, r["context"], r["projection_keep"], r["kv_keep"])
            costs[branch(r)][r["context"]].append(r["ms_per_token"] - ideal)
    return {b: {n: float(np.mean(v)) for n, v in sorted(by.items())} for b, by in costs.items()}


def dense_time(rows):
    by = defaultdict(list)
    for r in rows:
        if r["mode"] == "dense":
            by[r["context"]].append(r["ms_per_token"])
    ctx = sorted(by)
    lx, ly = np.log(ctx), np.log([np.mean(by[c]) for c in ctx])
    return lambda n: float(np.exp(np.interp(np.log(n), lx, ly)))


def interp(costs, n):
    return float(np.interp(n, list(costs), list(costs.values())))


def cost_fn(sweep, keep, key):
    if key in sweep:
        return lambda n: interp(sweep[key], n)
    # a keep ratio the sweep lacks: its 32K cost from the keep-ratio campaign, with the context
    # dependence of the swept cell of the same branch
    family = next(c for k, c in sweep.items() if k[0] == key[0])
    at32 = interp(keep[key], 32768)
    return lambda n: at32 + interp(family, n) - interp(family, 32768)


def predict(m, t_dense, c_proj, c_sel, rp, rkv):
    grid = np.arange(2048, 262144, 128)
    gap = np.array([t_dense(n) * (1 / bound(m, n, rp=rp) - 1 / bound(m, n, rkv=rkv)) + c_proj(n) - c_sel(n)
                    for n in grid])
    i = np.flatnonzero((gap[:-1] < 0) & (gap[1:] >= 0))
    if not len(i):
        return None
    i = i[0]
    return float(grid[i] + (grid[i + 1] - grid[i]) * -gap[i] / (gap[i + 1] - gap[i]))


def print_costs(costs, title):
    ctxs = sorted({n for c in costs.values() for n in c})
    print(title)
    print(f"{'branch':>16s} " + " ".join(f"{n // K:>6d}K" for n in ctxs))
    for (mode, rp, rkv), c in sorted(costs.items()):
        print(f"{mode:>5s} {rp:4.2f}/{rkv:4.2f} " + " ".join(f"{c[n]:7.2f}" if n in c else " " * 7 for n in ctxs))


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog,
                                 description="Kernel costs (Table 10) and cost-aware crossing predictions (Table 2).")
    ap.add_argument("sweep", type=Path, nargs="+", help="context-sweep campaign folders")
    ap.add_argument("--keep", type=Path, nargs="*", default=[], help="32K keep-ratio campaign folders")
    ap.add_argument("--dense", type=Path, nargs="*", default=None,
                    help="take the dense step time from another sweep, e.g. another GPU")
    ap.add_argument("--pairs", nargs="+", default=["0.7/0.3", "0.5/0.3", "0.5/0.5"])
    a = ap.parse_args(argv)
    rows = load_runs(a.sweep)
    m = model_for(rows[0]["model"])
    sweep = kernel_costs(rows, m)
    keep = kernel_costs(load_runs(a.keep), m) if a.keep else {}
    t_dense = dense_time(load_runs(a.dense) if a.dense else rows)

    print_costs(sweep, f"kernel cost beyond the byte time (ms), {rows[0]['model']} on {rows[0]['gpu']}")
    if keep:
        print_costs(keep, "\nkeep-ratio campaign")

    print(f"\n{'pair':>9s} {'n*':>8s} {'predicted':>10s} {'bandwidth':>10s} {'s (ms/K)':>9s}")
    for pair in a.pairs:
        rp, rkv = (float(x) for x in pair.split("/"))
        try:
            c_proj = cost_fn(sweep, keep, ("proj", rp, 1.0))
            c_sel = cost_fn(sweep, keep, ("sel", 1.0, rkv))
        except (KeyError, StopIteration):
            print(f"{pair:>9s} no kernel cost for this keep ratio, add a --keep campaign that measures it")
            continue
        x = predict(m, t_dense, c_proj, c_sel, rp, rkv)
        at = x or crossover(m, rp, rkv)
        bw = step_bytes(m, at) / t_dense(at)  # bytes per ms of dense decoding
        s = m.layers * kv_bytes(m, K) * (1 - rkv) / bw
        pred = f"{x / K:9.1f}K" if x else f"{'none':>10s}"
        print(f"{pair:>9s} {crossover(m, rp, rkv) / K:7.1f}K {pred} {bw / 1e9:6.2f} TB/s {s:9.3f}")


if __name__ == "__main__":
    main()

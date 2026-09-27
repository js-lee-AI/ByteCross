"""Table 2: byte crossover, cost-aware prediction and measured crossing. Tables 12 and 13 with --full."""

import argparse
from pathlib import Path

from bytecross.campaign import load_runs
from bytecross.crossing import analyze_group
from bytecross.predict import cost_fn, dense_time, kernel_costs, predict
from bytecross.traffic import K, crossover, model_for

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--records", type=Path, default=ROOT / "records")
ap.add_argument("--full", action="store_true", help="also print every context of the crossing campaigns")
a = ap.parse_args()


def runs(*folders):
    return load_runs([a.records / f for f in folders])


def predicted(sweep, keep, rp, rkv, dense=None):
    # costs come from the sweep and the 32K keep-ratio campaign only, never from a crossing cell
    rows = runs(sweep)
    m = model_for(rows[0]["model"])
    costs = kernel_costs(rows, m)
    keep_costs = kernel_costs(runs(keep), m) if keep else {}
    t_dense = dense_time(runs(dense) if dense else rows)
    return predict(m, t_dense, cost_fn(costs, keep_costs, ("proj", rp, 1.0)),
                   cost_fn(costs, keep_costs, ("sel", 1.0, rkv)), rp, rkv)


# GPU, model, keep pair, sweep, keep-ratio campaign, crossing campaign folders, group
# On RTX A6000 the 32K costs of keep ratios the sweep lacks come from the five-block re-timing at 32K.
SETUPS = [
    ("A100-SXM4", "8B", 0.7, 0.3, "sweep_a100", "keep_a100", ("crossing_a100_1", "crossing_a100_2"), "p70_s30"),
    ("A100-SXM4", "8B", 0.5, 0.3, "sweep_a100", "keep_a100", ("crossing_a100_1", "crossing_a100_2"), "p50_s30"),
    ("A100-SXM4", "8B", 0.5, 0.5, "sweep_a100", "keep_a100", ("crossing_a100_1", "crossing_a100_2"), "p50_s50"),
    ("A100-SXM4", "3B", 0.5, 0.3, "sweep_3b", None, ("crossing_3b",), "p50_s30"),
    ("RTX A6000", "8B", 0.7, 0.3, "sweep_a6000", "retime_a6000", ("crossing_a6000",), "p70_s30"),
    ("RTX A6000", "8B", 0.5, 0.3, "sweep_a6000", "retime_a6000", ("crossing_a6000",), "p50_s30"),
]

print("Table 2, in K tokens")
print(f"{'GPU':10s} {'model':5s} {'r_P/r_KV':>9s} {'n*':>7s} {'pred.':>7s} {'measured':>9s}  95% interval")
for gpu, size, rp, rkv, sweep, keep, folders, group in SETUPS:
    rows = runs(*folders)
    m = model_for(rows[0]["model"])
    x, ci = analyze_group(rows, group, show=False)
    pred = predicted(sweep, keep, rp, rkv)
    print(f"{gpu:10s} {size:5s} {rp:4.1f}/{rkv:.1f} {crossover(m, rp, rkv) / K:7.1f} {pred / K:7.1f} {x / K:9.1f}  "
          f"[{ci[0] / K:.1f}, {ci[1] / K:.1f}]")

print("\nAppendix D, A100-SXM4 kernel costs with the RTX A6000 dense step time")
for rp, rkv in [(0.7, 0.3), (0.5, 0.3)]:
    print(f"  {rp}/{rkv}  {predicted('sweep_a100', 'keep_a100', rp, rkv, dense='sweep_a6000') / K:.1f}K")

if a.full:
    print("\n== Table 12, A100-SXM4")
    for group in ("p70_s30", "p50_s30", "p50_s50"):
        analyze_group(runs("crossing_a100_1", "crossing_a100_2"), group)
    analyze_group(runs("crossing_3b"), "p50_s30")
    print("\n== Table 13, RTX A6000")
    for group in ("p70_s30", "p50_s30"):
        analyze_group(runs("crossing_a6000"), group)

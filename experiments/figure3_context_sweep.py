"""Figure 3 (speedup across context with byte bounds) and Table 10 (kernel costs), from the context sweeps."""

import argparse
from pathlib import Path

import numpy as np

from bytecross.campaign import load_runs, speedups, summary
from bytecross.predict import kernel_costs, print_costs
from bytecross.traffic import K, bound, model_for

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--records", type=Path, default=ROOT / "records")
a = ap.parse_args()

for folder in ("sweep_a100", "sweep_a6000"):
    rows = load_runs([a.records / folder])
    m = model_for(rows[0]["model"])
    print(f"\n== Figure 3, {folder}")
    summary(rows)
    keep = {r["cell"]: (r["projection_keep"], r["kv_keep"]) for r in rows}
    sp = speedups(rows)
    print("measured / byte bound, lowest and highest over contexts")
    for cell in ("proj50", "sel30", "p50_s30"):
        ratio = [np.mean(v) / bound(m, n, *keep[cell]) for n, v in sp[cell].items()]
        print(f"  {cell:8s} {min(ratio):.3f} to {max(ratio):.3f}")
    print_costs(kernel_costs(rows, m), f"\n== Table 10, kernel cost beyond the byte time (ms), {folder}")

rows = load_runs([a.records / "sweep_3b"])
print("\n== Llama-3.2-3B sweep (Appendix D)")
print_costs(kernel_costs(rows, model_for(rows[0]["model"])), "kernel cost beyond the byte time (ms)")

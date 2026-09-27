"""Figure 6: keep-ratio sweeps at 32K and the quality-matched crossover (Section 5.2)."""

import argparse
from pathlib import Path

from bytecross.campaign import load_runs, summary
from bytecross.traffic import K, MODELS, admissible_keep, load_quality_curves, matched_crossover

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--records", type=Path, default=ROOT / "records")
a = ap.parse_args()

print("== Figure 6 left, A100-SXM4 speedup at 32K with byte bounds")
summary(load_runs([a.records / "keep_a100"]))

Q = a.records / "quality"
proj, sel = load_quality_curves([Q / "grid_ppl.json", Q / "keep_extra_ppl.json"])
print("\n== Figure 6 middle, PPL increase at 32K")
print("projection keep " + "  ".join(f"{k}: {v:.3f}" for k, v in sorted(proj.items(), reverse=True)))
print("KV keep         " + "  ".join(f"{k}: {v:.3f}" for k, v in sorted(sel.items(), reverse=True)))

m = MODELS["llama-3.1-8b"]
print("\n== Figure 6 right, byte crossover at equal PPL increase")
print(f"{'budget':>6s} {'r_P':>6s} {'r_KV':>6s} {'n*':>8s}")
for b in (0.2, 0.5, 1.0):
    n = matched_crossover(m, proj, sel, b)
    print(f"{b:6.1f} {admissible_keep(proj, b):6.3f} {admissible_keep(sel, b):6.3f} {n / K:7.1f}K")

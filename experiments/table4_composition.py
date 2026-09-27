"""Tables 4, 16 and 17: composition under a perplexity budget on both GPUs."""

import argparse
from pathlib import Path

from bytecross import composition
from bytecross.campaign import load_runs, summary

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--records", type=Path, default=ROOT / "records")
a = ap.parse_args()
R, Q = a.records, a.records / "quality"

common = ["--ppl", str(Q / "grid_ppl.json"), "--screen", str(R / "screen_a6000"),
          "--retrieval", *map(str, sorted((Q / "retrieval").glob("*.json"))),
          # Table 16 bootstraps the three projection-only rows with their own seed
          "--projection-seed", "20260914"]
for gpu, folder in [("RTX A6000", "retime_a6000"), ("A100-SXM4", "retime_a100")]:
    print(f"\n== Table 16 (screen on RTX A6000) and Table 4, selected settings re-timed on {gpu}")
    composition.main(common + ["--retime", str(R / folder)])
    print(f"\n== Table 17, {gpu}")
    summary(load_runs([R / folder]))

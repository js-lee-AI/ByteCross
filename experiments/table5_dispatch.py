"""Table 5: dispatch policies priced on the request mix, from the context sweeps."""

import argparse
from pathlib import Path

from bytecross import dispatch

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--records", type=Path, default=ROOT / "records")
a = ap.parse_args()

# scoring pass over a 32K prompt, 0.24 s on A100-SXM4 and 0.32 s on RTX A6000 (Appendix A)
for gpu, folder, seconds in [("A100-SXM4", "sweep_a100", "0.24"), ("RTX A6000", "sweep_a6000", "0.32")]:
    print(f"\n== Table 5, {gpu} (the mixed row is the paper's request mix)")
    dispatch.main([str(a.records / folder), "--select-seconds", seconds])

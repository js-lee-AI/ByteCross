"""Table 3: perplexity increase at 32K and passkey and multi-key retrieval up to 127K."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--records", type=Path, default=ROOT / "records")
a = ap.parse_args()
Q = a.records / "quality"

ROWS = [("Dense", "dense", (1.0, 1.0)), ("Selection, 30%", "sel30", (1.0, 0.3)),
        ("Selection, 20%", "sel20", (1.0, 0.2)), ("Proj. 50% + sel. 20%", "p50_sel20", (0.5, 0.2))]
CONTEXTS = (32768, 65536, 130048)


def ppl_by_setting(path):
    out = {}
    for r in json.loads(path.read_text())["results"]:
        out[r["context"], r["projection_keep"], r["kv_keep"]] = r["ppl"]
    return out


def counts(test, name):
    summary = {}
    for folder in ("retrieval", "retrieval_long"):
        path = Q / folder / f"{test}_{name}.json"
        if path.exists():
            summary.update(json.loads(path.read_text())["summary"])
    return [f"{summary[str(n)]['correct']}/{summary[str(n)]['trials']}" for n in CONTEXTS]


grid = ppl_by_setting(Q / "grid_ppl.json")
print("Table 3 (Llama-3.1-8B-Instruct)")
print(f"{'configuration':22s} {'dPPL':>6s}   passkey 32K 64K 127K     multi-key 32K 64K 127K")
for label, name, (rp, rkv) in ROWS:
    d = grid[32768, rp, rkv] - grid[32768, 1.0, 1.0]
    dppl = "0" if name == "dense" else "<0.01" if abs(d) < 0.01 else f"{d:.2f}"
    print(f"{label:22s} {dppl:>6s}   {' '.join(counts('passkey', name)):22s}   {' '.join(counts('multikey', name))}")
print(f"dense perplexity at 32K {grid[32768, 1.0, 1.0]:.2f}")

long = ppl_by_setting(Q / "long_ppl.json")
print("\nDecode-position perplexity at 64K and 127K")
print(f"{'configuration':22s} {'64K':>7s} {'127K':>7s}   increase over dense")
for label, name, (rp, rkv) in ROWS:
    p = [long[n, rp, rkv] for n in CONTEXTS[1:]]
    d = [long[n, rp, rkv] - long[n, 1.0, 1.0] for n in CONTEXTS[1:]]
    print(f"{label:22s} {p[0]:7.3f} {p[1]:7.3f}   {d[0]:+.4f} {d[1]:+.4f}")

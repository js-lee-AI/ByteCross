"""Fresh-process timing campaigns in blocks, and the block-resampling statistics used throughout."""

import argparse
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bytecross.traffic import K, bound, model_for


def resamples(n):
    # every resample of n blocks with replacement (n**n rows)
    return np.array(list(itertools.product(range(n), repeat=n)))


def block_interval(values, q=(0.025, 0.975)):
    v = np.asarray(values, float)
    return np.quantile(v[resamples(len(v))].mean(axis=1), q)


def load_runs(dirs):
    rows, seen = [], set()
    for d in dirs:
        for path in sorted(Path(d).glob("*.json")):
            r = json.loads(path.read_text())
            key = (r["block"], r["group"], r["context"], r["cell"])
            if key in seen:
                raise ValueError(f"cell measured twice: {key}")
            seen.add(key)
            rows.append(r)
    gpus = {r["gpu"] for r in rows}
    if len(gpus) > 1:
        raise ValueError(f"one GPU model per analysis, found {gpus}")
    return rows


def speedups(rows):
    # paired with the dense cell of the same block, group and context
    dense = {(r["block"], r["group"], r["context"]): r["ms_per_token"] for r in rows if r["mode"] == "dense"}
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        base = dense.get((r["block"], r["group"], r["context"]))
        if base is not None:
            out[r["cell"]][r["context"]].append(base / r["ms_per_token"])
    return out


def run(spec, checkpoint, hist, out, blocks, max_context):
    out.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        for group in spec["groups"]:
            ctxs = [c for c in group["contexts"] if max_context is None or c <= max_context]
            if b % 2:
                ctxs = ctxs[::-1]
            r = b % len(group["cells"])
            cells = group["cells"][r:] + group["cells"][:r]
            for ctx in ctxs:
                for cell in cells:
                    path = out / f"b{b}_{group['name']}_{ctx}_{cell['id']}.json"
                    if path.exists():
                        continue
                    meta = {"block": b, "group": group["name"], "cell": cell["id"]}
                    cmd = [sys.executable, "-m", "bytecross.timing", "--mode", cell["mode"], "--context", str(ctx),
                           "--projection-keep", str(cell.get("projection_keep", 1.0)),
                           "--kv-keep", str(cell.get("kv_keep", 1.0)), "--checkpoint", str(checkpoint),
                           "--meta", json.dumps(meta), "--output", str(path)]
                    if hist:
                        cmd += ["--hist", str(hist)]
                    if "keep_tokens" in cell:
                        cmd += ["--keep-tokens", str(cell["keep_tokens"])]
                    print(" ".join(cmd[2:]), flush=True)
                    subprocess.run(cmd, check=True)


def summary(rows):
    m = model_for(rows[0]["model"])
    cells = {r["cell"]: r for r in rows}
    tps = defaultdict(list)
    for r in rows:
        tps[r["cell"], r["context"]].append(r["tokens_per_sec"])
    print(f"{rows[0]['model']} on {rows[0]['gpu']}")
    print(f"{'cell':10s} {'context':>8s} {'blocks':>6s} {'tok/s':>8s} {'speedup':>8s} {'95% interval':>17s} "
          f"{'bound':>6s} {'removed':>8s}")
    for cell, by in speedups(rows).items():
        c = cells[cell]
        for n, v in sorted(by.items()):
            b = bound(m, n, c["projection_keep"], c["kv_keep"])
            ci = "[{:.3f}, {:.3f}]".format(*block_interval(v)) if len(v) > 1 else ""
            print(f"{cell:10s} {n // K:7d}K {len(v):6d} {np.mean(tps[cell, n]):8.2f} {np.mean(v):8.3f} {ci:>17s} "
                  f"{b:6.3f} {1 - 1 / b:8.3f}")


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description="Fresh-process timing campaigns in blocks.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="time every cell of a spec, one fresh process per cell (GPU)")
    r.add_argument("spec", type=Path)
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--hist", type=Path, default=None)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--blocks", type=int, nargs="+", default=None)
    r.add_argument("--max-context", type=int, default=None)
    s = sub.add_parser("summary", help="speedup over dense with block intervals and byte bounds (CPU)")
    s.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        spec = json.loads(a.spec.read_text())
        blocks = a.blocks if a.blocks is not None else range(spec["blocks"])
        run(spec, a.checkpoint, a.hist, a.out, blocks, a.max_context)
    else:
        summary(load_runs(a.runs))


if __name__ == "__main__":
    main()

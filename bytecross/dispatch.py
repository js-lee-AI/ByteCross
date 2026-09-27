"""Dispatch policies priced on synthetic request mixes (Table 5)."""

import argparse
import math
import random
from pathlib import Path

import numpy as np

from bytecross.campaign import load_runs

MODES = ("dense", "proj", "sel", "both")


def workloads(seed=0):
    rng = random.Random(seed)
    chat = [(int(min(32768, max(64, rng.lognormvariate(math.log(1024), 1.1)))), 256) for _ in range(2000)]
    long_doc = [(rng.randint(32768, 131072), 128) for _ in range(500)]
    agent = [(rng.randint(16384, 98304), 1024) for _ in range(500)]
    return {"chat": chat, "long_doc": long_doc, "agent": agent,
            "mixed": chat[:1400] + long_doc[:400] + agent[:200]}


class Prices:
    def __init__(self, rows, select_seconds):
        # every measured cell of every block is one point of the price curve, as for Table 5
        self.tab = {}
        for mode in MODES:
            pts = sorted((r["context"], r["ms_per_token"]) for r in rows if r["mode"] == mode)
            self.tab[mode] = (np.log([c for c, _ in pts]), np.log([t for _, t in pts]))
        self.select_ms = 1000.0 * select_seconds

    def step(self, mode, n):
        # log-log interpolation, clamped at both measured ends
        lx, ly = self.tab[mode]
        x = math.log(max(n, 1))
        y = ly[0] if x <= lx[0] else np.interp(x, lx, ly)
        return float(np.exp(y))

    def request(self, mode, prompt, answer, seg=64):
        total, t = 0.0, 0
        while t < answer:
            k = min(seg, answer - t)
            total += k * self.step(mode, prompt + t + k // 2)
            t += k
        if mode in ("sel", "both"):
            total += self.select_ms * prompt / 32768  # one scoring pass over the prompt
        return total


def policy_speedups(prices, reqs):
    policies = {
        "always projection": lambda p, g: "proj",
        "always selection": lambda p, g: "sel",
        "always both": lambda p, g: "both",
        "calibrated gate": lambda p, g: min(MODES, key=lambda m: prices.request(m, p, g)),
    }
    dense = sum(prices.request("dense", p, g) for p, g in reqs)
    return {name: dense / sum(prices.request(choose(p, g), p, g) for p, g in reqs)
            for name, choose in policies.items()}


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description="Price dispatch policies on synthetic request mixes (Table 5).")
    ap.add_argument("sweep", type=Path, nargs="+", help="context-sweep campaign folders of one GPU")
    ap.add_argument("--select-seconds", type=float, required=True,
                    help="scoring and gather time of a 32K prompt, e.g. 0.24 on A100-SXM4, 0.32 on RTX A6000")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    prices = Prices(load_runs(a.sweep), a.select_seconds)
    rows = {name: policy_speedups(prices, reqs) for name, reqs in workloads(a.seed).items()}
    print(f"{'workload':>9s} " + " ".join(f"{p:>18s}" for p in next(iter(rows.values()))))
    for name, cells in rows.items():
        print(f"{name:>9s} " + " ".join(f"{c:18.3f}" for c in cells.values()))


if __name__ == "__main__":
    main()

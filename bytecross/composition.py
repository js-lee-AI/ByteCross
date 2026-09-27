"""Composition under a perplexity budget: the grid (Table 16), budget selection (Table 4), re-timing (Table 17)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bytecross.campaign import load_runs, resamples


def setting(r):
    return r["projection_keep"], r["kv_keep"]


def label(s):
    return f"{s[0]:.1f}/{s[1]:.1f}"


def load_ppl(paths, context):
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["results"]:
            if r["context"] == context:
                out[setting(r)] = r
    return out


def load_retrieval(paths, context):
    ok = defaultdict(lambda: True)
    for p in paths:
        d = json.loads(Path(p).read_text())
        s = d["summary"].get(str(context))
        if s is None:
            continue  # a run at other contexts
        ok[d["projection_keep"], d["kv_keep"]] &= s["correct"] == s["trials"]
    return ok


def ms_by_block(rows):
    return {(r["block"], setting(r)): r["ms_per_token"] for r in rows}


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog,
                                 description="Composition grid (Table 16), budget selection and timing (Tables 4, 17).")
    ap.add_argument("--ppl", type=Path, nargs="+", required=True, help="ppl.py outputs covering the grid and dense")
    ap.add_argument("--screen", type=Path, nargs="+", required=True, help="one-block timing screen of the grid")
    ap.add_argument("--retime", type=Path, nargs="*", default=[], help="independent blocks of the selected settings")
    ap.add_argument("--retrieval", type=Path, nargs="*", default=[], help="retrieval.py outputs; a setting must pass all")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--projection-seed", type=int, default=None,
                    help="own seed for the projection-only rows (Table 16 uses 20260914)")
    a = ap.parse_args(argv)

    ppl = load_ppl(a.ppl, a.context)
    dense = ppl[1.0, 1.0]
    starts = [w["start"] for w in dense["windows"]]
    tok = np.array([w["tokens"] for w in dense["windows"]])
    d_nll = np.array([w["nll"] for w in dense["windows"]])
    # paired resamples of the windows, shared by every setting
    idx = np.random.default_rng(a.seed).integers(0, len(tok), (a.draws, len(tok)))
    idx_proj = idx if a.projection_seed is None else \
        np.random.default_rng(a.projection_seed).integers(0, len(tok), (a.draws, len(tok)))
    screen = {setting(r): r["tokens_per_sec"] for r in load_runs(a.screen)}
    ok = load_retrieval(a.retrieval, a.context)

    print(f"{'r_P/r_KV':>8s} {'PPL':>7s} {'increase':>8s} {'95% interval':>17s} {'speedup':>8s} {'retrieval':>9s}")
    delta = {}
    for s in sorted(ppl, key=lambda s: (s[0] < 1 and s[1] < 1, s[1] < 1 and s[0] == 1, -s[0], -s[1])):
        r = ppl[s]
        assert [w["start"] for w in r["windows"]] == starts, "windows differ from the dense run"
        nll = np.array([w["nll"] for w in r["windows"]])
        ix = idx_proj if s[1] == 1.0 else idx
        boot = np.exp(nll[ix].sum(1) / tok[ix].sum(1)) - np.exp(d_nll[ix].sum(1) / tok[ix].sum(1))
        delta[s] = r["ppl"] - dense["ppl"]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        print(f"{label(s):>8s} {r['ppl']:7.3f} {delta[s]:8.3f} [{lo:6.3f}, {hi:6.3f}] "
              f"{screen[s] / screen[1.0, 1.0]:8.3f} {'pass' if ok[s] else 'fail':>9s}")

    ms = ms_by_block(load_runs(a.retime)) if a.retime else {}
    blocks = sorted({b for b, _ in ms})

    def retimed(s):
        return bool(ms) and all((b, s) in ms for b in blocks)

    def per_block(s):
        return np.array([ms[b, (1.0, 1.0)] / ms[b, s] for b in blocks])

    def retimed_speedup(s):
        return per_block(s).mean()

    print(f"\n{'budget':>6s} {'fastest':>8s} {'single':>8s} {'dPPL':>6s} {'vs dense':>9s} {'vs single [95%]':>24s} {'min':>6s}")
    for budget in a.budgets:
        eligible = [s for s in ppl if s != (1.0, 1.0) and delta[s] <= budget and ok[s]]
        if not eligible:
            print(f"{budget:6.2f} no eligible setting")
            continue
        single = [s for s in eligible if 1.0 in s]
        win = max(eligible, key=screen.get)
        # the comparator is the fastest eligible single branch, judged on the re-timed GPU if there is one
        timed = [s for s in single if retimed(s)]
        comp = max(timed, key=retimed_speedup) if timed else max(single, key=screen.get) if single else None
        line = f"{budget:6.2f} {label(win):>8s} {label(comp) if comp else '-':>8s} {delta[win]:6.3f}"
        missing = [x for x in (win, comp) if x and not retimed(x)]
        if ms and missing:
            line += " not re-timed: " + ", ".join(map(label, missing))
        elif ms:
            line += f" {retimed_speedup(win):8.3f}x"
            if comp and comp != win:
                # ratio of the two mean speedups over resampled blocks, and the smallest per-block gain
                sw, sc = per_block(win), per_block(comp)
                idx = resamples(len(blocks))
                lo, hi = np.quantile(sw[idx].mean(1) / sc[idx].mean(1), [0.025, 0.975])
                line += f" {sw.mean() / sc.mean():8.3f} [{lo:.3f}, {hi:.3f}] {(sw / sc).min():6.3f}"
        else:
            line += f" {screen[win] / screen[1.0, 1.0]:8.3f}x (screen)"
        print(line)


if __name__ == "__main__":
    main()

"""The `bytecross` command. `python -m bytecross` runs the same thing."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module

from . import __version__

# Subcommands that hand their arguments to the main() of a module.
TOOLS = {
    "traffic": "byte account tables (Tables 1 and 9), byte bounds, quality-matched crossover",
    "campaign": "run a timing campaign on a GPU, or summarize one on CPU",
    "crossing": "latency crossings of crossing campaigns with block intervals (Tables 12 and 13)",
    "predict": "kernel costs and cost-aware crossing predictions (Tables 2 and 10)",
    "composition": "composition under a perplexity budget (Tables 4, 16 and 17)",
    "dispatch": "dispatch policies priced on request mixes (Table 5)",
}


def _demo(args):
    from .traffic import MODELS, bound, crossover

    m = MODELS["llama-3.1-8b"]
    for rp, rkv in [(0.7, 0.3), (0.5, 0.3), (0.5, 0.5)]:
        print(f"keep {rp} / {rkv}   n* = {crossover(m, rp, rkv) / 1024:.1f}K tokens")
    print(f"KV keep 0.3 at 128K   speedup bound {bound(m, 131072, rkv=0.3):.2f}x")
    return 0


def _crossover(args):
    from .traffic import MODELS, Model, bound, crossover

    m = Model.from_config(args.config) if args.config else MODELS[args.model]
    name = args.config or args.model
    print(f"{name}: {m.layers} layers, d={m.d}, d_ff={m.d_ff}, {m.kv_heads} KV heads of {m.head_dim}")
    print(f"byte crossover at keep {args.rp} / {args.rkv}   n* = {crossover(m, args.rp, args.rkv) / 1024:.1f}K "
          f"(feed-forward projections only {crossover(m, args.rp, args.rkv, ff_only=True) / 1024:.1f}K)")
    print(f"{'context':>8s} {'proj':>6s} {'KV':>6s} {'both':>6s}   ideal speedup over dense")
    for n in args.contexts:
        print(f"{n // 1024:7d}K {bound(m, n, args.rp):6.3f} {bound(m, n, rkv=args.rkv):6.3f} "
              f"{bound(m, n, args.rp, args.rkv):6.3f}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        prog="bytecross",
        description="Byte crossover of activation sparsity and KV-cache sparsity in LLM decoding.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart")
    demo.set_defaults(func=_demo)

    cross = sub.add_parser("crossover", help="byte crossover and speedup bounds for one model")
    src = cross.add_mutually_exclusive_group()
    src.add_argument("--model", default="llama-3.1-8b",
                     choices=("qwen3-8b", "llama-3.1-8b", "mistral-7b", "llama-3.1-70b", "llama-3.2-3b"))
    src.add_argument("--config", help="a Hugging Face config.json instead of a built-in model")
    cross.add_argument("--rp", type=float, default=0.5, help="projection keep ratio")
    cross.add_argument("--rkv", type=float, default=0.3, help="KV keep ratio")
    cross.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 32768, 131072])
    cross.set_defaults(func=_crossover)

    for name, text in TOOLS.items():
        sub.add_parser(name, help=text, add_help=False)

    # everything after a tool name belongs to that tool's own parser
    if argv and argv[0] in TOOLS:
        import_module(f".{argv[0]}", __package__).main(argv[1:], prog=f"bytecross {argv[0]}")
        return 0
    args = parser.parse_args(argv)
    return args.func(args)

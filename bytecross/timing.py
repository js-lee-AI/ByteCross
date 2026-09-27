"""One timing cell: one mode at one context, in a fresh process."""

import argparse
import json
import time
from pathlib import Path

import torch

from bytecross import harness, kvselect

WARMUP, STEPS, REPEATS = 5, 50, 5


@torch.no_grad()
def time_decode(model, token, pos):
    torch._dynamo.reset()
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True
    step = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True, dynamic=False)
    token = torch.tensor([[token]], device="cuda", dtype=torch.int)

    def decode(steps):
        nonlocal token, pos
        for _ in range(steps):
            logits = step(token, torch.tensor([pos], device="cuda", dtype=torch.int))
            token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            pos += 1

    decode(WARMUP)
    tps = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode(STEPS)
        torch.cuda.synchronize()
        tps.append(STEPS / (time.perf_counter() - t0))
    return tps


def main():
    ap = argparse.ArgumentParser(description="One timing cell: one mode at one context, in a fresh process.")
    ap.add_argument("--mode", choices=("dense", "proj", "sel", "both"), required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--projection-keep", type=float, default=0.5)
    ap.add_argument("--kv-keep", type=float, default=0.3)
    ap.add_argument("--keep-tokens", type=int, default=None, help="exact KV budget; overrides --kv-keep")
    ap.add_argument("--checkpoint", type=Path, required=True, help="gpt-fast model.pth")
    ap.add_argument("--hist", type=Path, default=None, help="TEAL histograms (default: Llama-3-8B)")
    ap.add_argument("--meta", default="{}", help="JSON merged into the output")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("run each cell with exactly one visible GPU")
    rp = a.projection_keep if a.mode in ("proj", "both") else 1.0
    rkv = a.kv_keep if a.mode in ("sel", "both") else 1.0
    n, gen = a.context, WARMUP + STEPS * REPEATS

    tok = harness.tokenizer(a.checkpoint)
    stream = harness.wikitext_ids(tok)
    if stream.numel() <= n:
        raise SystemExit(f"WikiText-2 has {stream.numel()} tokens, {n + 1} needed")
    model = harness.load_model(a.checkpoint, n + gen + 32)
    if rp < 1:
        harness.sparsify_projections(model, a.hist, rp)
    harness.install(model, harness.splitk_forward)
    with torch.device("cuda"):
        model.setup_caches(1, n + gen + 16)
    # real-text cache state: activation sparsity depends on what the cache holds
    prefill_s = harness.prefill(model, stream[:n].cuda(), chunk=512, flash=True)
    budget = static = gather_s = None
    if rkv < 1:
        budget = a.keep_tokens or max(1, int(model.max_seq_length * rkv))
        static = budget - gen  # with the decoded tokens, never more entries than the budget
        gather_s = kvselect.random_compact(model, n, static, gen)
    tps = time_decode(model, int(stream[n]), n)

    mean = sum(tps) / len(tps)
    out = {"mode": a.mode, "context": n, "projection_keep": rp, "kv_keep": rkv, "keep_tokens": budget,
           "static_tokens": static, "allocated_tokens": model.max_seq_length, "tokens_per_sec": mean,
           "tokens_per_sec_runs": tps, "ms_per_token": 1000.0 / mean, "prefill_seconds": prefill_s,
           "gather_seconds": gather_s, "model": a.checkpoint.parent.name, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, **json.loads(a.meta)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1))
    print(f"{a.mode} ctx={n} {mean:.2f} tok/s {1000 / mean:.3f} ms/token", flush=True)


if __name__ == "__main__":
    main()

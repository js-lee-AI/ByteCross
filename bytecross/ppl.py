"""Decode-position perplexity on WikiText-2."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from bytecross import harness, kvselect


def window_starts(total, length, count):
    if count == 1 or total == length:
        return [0]
    stride = max(1, (total - length) // (count - 1))
    return [i * stride for i in range(count)]


@torch.no_grad()
def score(model, tokens, first):
    # teacher-forced single-token decode steps predicting tokens[first:]
    nll = 0.0
    for pos in range(first - 1, tokens.numel() - 1):
        logits = model(tokens[pos].view(1, 1), torch.tensor([pos], device=tokens.device, dtype=torch.int))
        nll -= torch.log_softmax(logits[0, -1].float(), dim=-1)[tokens[pos + 1]].item()
    return nll, tokens.numel() - first


def main():
    ap = argparse.ArgumentParser(description="Decode-position perplexity on WikiText-2.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--hist", type=Path, default=None)
    ap.add_argument("--settings", nargs="+", default=["1.0/1.0"],
                    help="projection keep / KV keep; 1.0 disables a branch")
    ap.add_argument("--contexts", type=int, nargs="+", default=[32768])
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--scored", type=int, default=512)
    ap.add_argument("--flash-prefill", action="store_true",
                    help="FlashAttention prefill and a lazy causal mask (needed from 64K)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    stream = harness.wikitext_ids(harness.tokenizer(a.checkpoint)).cuda()
    results = []
    for length in a.contexts:
        starts = window_starts(stream.numel(), length, a.windows)
        first = length - a.scored
        for setting in a.settings:
            rp, rkv = harness.parse_setting(setting)
            model = harness.load_for_eval(a.checkpoint, a.hist, rp, length + 16, a.flash_prefill)
            budget = int(model.max_seq_length * rkv) if rkv < 1 else None
            windows = []
            for s0 in starts:
                tokens = stream[s0:s0 + length]
                harness.prefill(model, tokens[:first], a.chunk, flash=a.flash_prefill)
                row = {"start": s0}
                if budget:
                    # the selection plus the scored tokens never exceed the budget
                    row["score_seconds"], row["gather_seconds"] = kvselect.select_and_compact(
                        model, tokens, first - 1, budget - a.scored, a.scored)
                row["nll"], row["tokens"] = score(model, tokens, first)
                windows.append(row)
                print(f"L={length} {setting} start={s0} ppl={math.exp(row['nll'] / row['tokens']):.4f}", flush=True)
            nll, n = sum(w["nll"] for w in windows), sum(w["tokens"] for w in windows)
            results.append({"context": length, "projection_keep": rp, "kv_keep": rkv, "keep_tokens": budget,
                            "ppl": math.exp(nll / n), "nll": nll, "tokens": n, "windows": windows})
            print(f"L={length} {setting} PPL={math.exp(nll / n):.4f}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps({"results": results}, indent=1))


if __name__ == "__main__":
    main()

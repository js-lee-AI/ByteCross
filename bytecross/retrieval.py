"""Passkey and multi-key retrieval under one setting."""

import argparse
import gc
import json
import random
import re
from pathlib import Path

import torch

from bytecross import harness, kvselect

USER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

PASSKEY_PREFIX = USER + "Memorize the six-digit access code embedded below. "
PASSKEY_FILLER = ("This passage discusses memory bandwidth, language model inference, "
                  "scheduling, and systems evaluation without mentioning any digits. ")
PASSKEY_NEEDLE = "\nThe access code is {code}. Keep it for the final question.\n"
PASSKEY_SUFFIX = "\nWhat is the six-digit access code? Reply with the code only." + ASSISTANT

LABELS = ["amber", "cobalt", "crimson", "ivory", "jade", "onyx", "scarlet", "violet"]
MULTI_PREFIX = USER + ("The text below contains several access codes, each tied to a named archive. "
                       "Answer the question at the end.\n\n")
MULTI_NEEDLE = "\nThe access code for the {label} archive is {code}.\n"
MULTI_SUFFIX = "\n\nWhat is the access code for the {label} archive? Reply with the code only." + ASSISTANT

DEPTHS = (0.05, 0.25, 0.5, 0.75, 0.95)


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def passkey_prompts(tok, context, seed, trials):
    prefix, filler, suffix = encode(tok, PASSKEY_PREFIX), encode(tok, PASSKEY_FILLER), encode(tok, PASSKEY_SUFFIX)
    rng = random.Random(seed + context)
    for depth in DEPTHS:
        for _ in range(trials):
            code = str(rng.randrange(100000, 1000000))
            needle = encode(tok, PASSKEY_NEEDLE.format(code=code))
            budget = context - len(prefix) - len(needle) - len(suffix)
            before = max(0, min(budget, round(budget * depth)))
            fill = filler * (budget // len(filler) + 1)
            yield prefix + fill[:before] + needle + fill[:budget - before] + suffix, code, [], depth


def multikey_prompts(tok, stream, context, seed, trials, keys=4):
    rng = random.Random(seed + context)
    prefix = encode(tok, MULTI_PREFIX)
    for depth in DEPTHS:
        for _ in range(trials):
            labels = rng.sample(LABELS, keys)
            codes = [str(rng.randrange(100000, 1000000)) for _ in labels]
            depths = [depth]
            while len(depths) < keys:
                f = rng.uniform(0.02, 0.98)
                if all(abs(f - g) >= 0.05 for g in depths):
                    depths.append(f)
            offset = rng.randrange(0, len(stream) - context)
            hay = stream[offset:offset + context]
            suffix = encode(tok, MULTI_SUFFIX.format(label=labels[0]))
            needles = sorted((d, encode(tok, MULTI_NEEDLE.format(label=l, code=c)))
                             for l, c, d in zip(labels, codes, depths))
            budget = context - len(prefix) - len(suffix) - sum(len(ids) for _, ids in needles)
            ids, used = list(prefix), 0
            for d, needle in needles:
                cut = max(used, min(budget, round(budget * d)))
                ids += hay[used:cut] + needle
                used = cut
            ids += hay[used:budget] + suffix
            assert len(ids) == context
            yield ids, codes[0], codes[1:], depth


@torch.no_grad()
def greedy(model, prompt, steps, tok):
    token, pos, out = prompt[-1:].view(1, 1), prompt.numel() - 1, []
    for i in range(steps):
        logits = model(token, torch.tensor([pos + i], device=prompt.device, dtype=torch.int))
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        out.append(int(token))
    return tok.decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description="Passkey and multi-key retrieval under one setting.")
    ap.add_argument("test", choices=("passkey", "multikey"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--hist", type=Path, default=None)
    ap.add_argument("--setting", default="1.0/1.0", help="projection keep / KV keep; 1.0 disables a branch")
    ap.add_argument("--contexts", type=int, nargs="+", default=[32768])
    ap.add_argument("--trials", type=int, default=8, help="per depth")
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--flash-prefill", action="store_true",
                    help="FlashAttention prefill and a lazy causal mask (needed from 64K)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else {"passkey": 20260903, "multikey": 20260925}[a.test]
    rp, rkv = harness.parse_setting(a.setting)
    tok = harness.tokenizer(a.checkpoint)
    stream = harness.wikitext_ids(tok).tolist() if a.test == "multikey" else None
    trials, summary = [], {}
    for context in a.contexts:
        model = harness.load_for_eval(a.checkpoint, a.hist, rp, context + a.new_tokens + 16, a.flash_prefill)
        budget = int(model.max_seq_length * rkv) if rkv < 1 else None
        prompts = (passkey_prompts(tok, context, seed, a.trials) if a.test == "passkey"
                   else multikey_prompts(tok, stream, context, seed, a.trials))
        rows = []
        for ids, code, distractors, depth in prompts:
            prompt = torch.tensor(ids, device="cuda", dtype=torch.int)
            harness.prefill(model, prompt, a.chunk, flash=a.flash_prefill)
            if budget:
                kvselect.select_and_compact(model, prompt, prompt.numel() - 1, budget - a.new_tokens, a.new_tokens)
            text = greedy(model, prompt, a.new_tokens, tok)
            m = re.search(r"(?<!\d)\d{6}(?!\d)", text)
            pred = m.group(0) if m else None
            rows.append({"context": context, "depth": depth, "code": code, "prediction": pred, "answer": text,
                         "correct": pred == code, "distractor": pred in distractors})
            print(f"{a.test} ctx={context} depth={depth} correct={pred == code} answer={text!r}", flush=True)
        summary[context] = {"correct": sum(r["correct"] for r in rows), "trials": len(rows),
                            "distractor_answers": sum(r["distractor"] for r in rows), "keep_tokens": budget}
        print(context, summary[context], flush=True)
        trials += rows
        del model
        gc.collect()
        torch.cuda.empty_cache()
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({"test": a.test, "projection_keep": rp, "kv_keep": rkv, "seed": seed,
                                        "summary": summary, "trials": trials}, indent=1))


if __name__ == "__main__":
    main()

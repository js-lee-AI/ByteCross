"""Convert a Hugging Face Llama 3.x snapshot to the gpt-fast checkpoint layout."""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

# Key map and q/k permutation as in gpt-fast's scripts/convert_hf_checkpoint.py (BSD-3-Clause).
KEYS = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}
TOKENIZER_FILES = ("config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="Convert a Hugging Face Llama 3.x snapshot to the gpt-fast layout.")
    ap.add_argument("--source", type=Path, required=True, help="local Hugging Face snapshot folder")
    ap.add_argument("--dest", type=Path, required=True,
                    help="output folder named after the model, e.g. checkpoints/Llama-3.1-8B-Instruct")
    a = ap.parse_args()
    cfg = json.loads((a.source / "config.json").read_text())
    n_head, n_kv, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    head_dim = dim // n_head

    state = {}
    for f in sorted(a.source.glob("*.safetensors")):
        state.update(load_file(str(f), device="cpu"))
    out = {}
    for key, value in state.items():
        layer = re.search(r"\d+", key)
        out[KEYS[re.sub(r"\d+", "{}", key)].format(layer.group(0)) if layer else KEYS[key]] = value
    if "output.weight" not in out:
        assert cfg.get("tie_word_embeddings"), "untied checkpoint without an output head"
        out["output.weight"] = out["tok_embeddings.weight"].clone()

    def permute(w, heads):
        return w.view(heads, 2, head_dim // 2, dim).transpose(1, 2).reshape(heads * head_dim, dim)

    for key in [k for k in out if ".wq." in k]:
        q, k, v = out.pop(key), out.pop(key.replace("wq", "wk")), out.pop(key.replace("wq", "wv"))
        out[key.replace("wq", "wqkv")] = torch.cat([permute(q, n_head), permute(k, n_kv), v])
    a.dest.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.dest / "model.pth")
    for name in TOKENIZER_FILES:
        if (a.source / name).exists():
            shutil.copy(a.source / name, a.dest / name)
    print(f"saved {a.dest / 'model.pth'}")


if __name__ == "__main__":
    main()

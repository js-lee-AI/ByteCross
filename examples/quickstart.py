"""The byte crossover of Llama-3.1-8B from its config. CPU only, no downloads."""

import bytecross

# the fields of the Llama-3.1-8B-Instruct config.json that the byte account reads
config = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "vocab_size": 128256,
          "num_attention_heads": 32, "num_key_value_heads": 8, "max_position_embeddings": 131072}
m = bytecross.Model.from_config(config)

for rp, rkv in [(0.7, 0.3), (0.5, 0.3), (0.5, 0.5)]:
    print(f"keep {rp} / {rkv}   n* = {bytecross.crossover(m, rp, rkv) / 1024:.1f}K tokens")
print(f"KV keep 0.3 at 128K   speedup bound {bytecross.bound(m, 131072, rkv=0.3):.2f}x")

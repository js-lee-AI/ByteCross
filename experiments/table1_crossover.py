"""Table 1 and the byte counts of Section 3, from model dimensions alone. CPU only."""

from bytecross.traffic import K, MODELS, Model, attn_bytes, crossover, kv_bytes, mlp_bytes

print("Table 1: byte crossover at 50% projection keep and 30% KV keep")
print(f"{'model':14s} {'d':>5s} {'d_ff':>6s} {'n*':>8s} {'n*_FF':>8s} {'window':>7s}")
for name in ("qwen3-8b", "llama-3.1-8b", "mistral-7b", "llama-3.1-70b"):
    m = MODELS[name]
    print(f"{name:14s} {m.d:5d} {m.d_ff:6d} {crossover(m, 0.5, 0.3) / K:7.1f}K "
          f"{crossover(m, 0.5, 0.3, ff_only=True) / K:7.1f}K {m.window // K:6d}K")

m = MODELS["llama-3.1-8b"]
print("\nSection 3.1, one Llama-3.1-8B layer")
print(f"projection weights read at every step   {(mlp_bytes(m) + attn_bytes(m)) / 1e6:.0f} MB")
print(f"cache bytes per token                   {kv_bytes(m, 1)}")
print(f"cache at n=512 and at n=32K             {kv_bytes(m, 512) / 1e6:.0f} MB and {kv_bytes(m, 32768) / 1e6:.0f} MB")

m32 = Model(m.layers, m.d, m.d_ff, m.vocab, kv_heads=32, window=m.window)
print("\nSection 3.3")
print(f"Llama-3.1-8B with 32 KV heads           n* = {crossover(m32, 0.5, 0.3) / K:.1f}K")
ratio = crossover(m, 0.5, 0.3, weight_bytes=0.5) / crossover(m, 0.5, 0.3)
print(f"4-bit weights with a 16-bit cache       n* scales by {ratio}")

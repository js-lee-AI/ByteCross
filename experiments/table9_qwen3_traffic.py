"""Table 9, the Qwen3-8B decode step in BF16 from its dimensions. CPU only."""

from bytecross.traffic import MODELS, other_bytes, print_step_shares

print("Table 9: share of step reads (%) and MB saved by the FF-only branch at 50% keep and KV at 30% keep")
print_step_shares("qwen3-8b", ff_keep=0.5, rkv=0.3)
print(f"\nOther (head, one embedding row, norms) {other_bytes(MODELS['qwen3-8b']) / 1e9:.3f} GB")

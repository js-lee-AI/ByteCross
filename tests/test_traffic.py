from bytecross.traffic import (K, MODELS, Model, attn_bytes, bound, crossover, crossover_batch,
                               crossover_fixed_budget, kv_bytes, matched_crossover, mlp_bytes, other_bytes,
                               step_bytes)

LLAMA8B = MODELS["llama-3.1-8b"]


def test_table1():
    # model: (n*, n*_FF) in K tokens at r_P = 0.5, r_KV = 0.3
    table = {"qwen3-8b": (65.7, 51.4), "llama-3.1-8b": (74.3, 60.0),
             "mistral-7b": (74.3, 60.0), "llama-3.1-70b": (291.4, 240.0)}
    for name, (full, ff) in table.items():
        m = MODELS[name]
        assert round(crossover(m, 0.5, 0.3) / K, 1) == full, name
        assert round(crossover(m, 0.5, 0.3, ff_only=True) / K, 1) == ff, name


def test_crossover_tokens():
    assert round(crossover(LLAMA8B, 0.5, 0.3)) == 76069
    assert round(crossover(MODELS["mistral-7b"], 0.5, 0.3)) == 76069
    assert round(crossover(MODELS["qwen3-8b"], 0.5, 0.3)) == 67291
    assert round(crossover(LLAMA8B, 0.5, 0.3, ff_only=True)) == 61440
    assert round(crossover(MODELS["qwen3-8b"], 0.5, 0.3, ff_only=True)) == 52663
    assert round(crossover(LLAMA8B, 0.7, 0.3)) == 45641
    assert round(crossover(LLAMA8B, 0.5, 0.5)) == 106496
    assert round(crossover(MODELS["llama-3.2-3b"], 0.5, 0.3)) == 35109
    # Table 2 byte crossovers
    for (rp, rkv), n in {(0.7, 0.3): 44.6, (0.5, 0.3): 74.3, (0.5, 0.5): 104.0}.items():
        assert round(crossover(LLAMA8B, rp, rkv) / K, 1) == n
    assert round(crossover(MODELS["llama-3.2-3b"], 0.5, 0.3) / K, 1) == 34.3


def test_crossover_variants():
    m32 = Model(32, 4096, 14336, 128256, kv_heads=32)
    assert round(crossover(m32, 0.5, 0.3) / K, 1) == 20.7
    # 4-bit weights with a 16-bit cache: four times closer
    assert abs(crossover(LLAMA8B, 0.5, 0.3, weight_bytes=0.5) - crossover(LLAMA8B, 0.5, 0.3) / 4) < 1e-6
    n = crossover_fixed_budget(LLAMA8B, 0.5, 4096)
    assert abs(crossover(LLAMA8B, 0.5, 4096 / n) - n) < 1e-6
    assert crossover_batch(LLAMA8B, 0.5, 0.3, 1) == crossover(LLAMA8B, 0.5, 0.3)
    assert round(crossover_batch(LLAMA8B, 0.92, 0.3, 4) / K, 1) == 3.0


def test_layer_bytes():
    # one Llama-3.1-8B layer: 436 MB of projection weights, 4096 n bytes of cache
    assert round((mlp_bytes(LLAMA8B) + attn_bytes(LLAMA8B)) / 1e6) == 436
    assert kv_bytes(LLAMA8B, 1) == 4096


def test_qwen_step_shares():
    m = MODELS["qwen3-8b"]
    assert round(other_bytes(m) / 1e9, 3) == 1.245
    expected = {512: (71.5, 19.9, 0.5, 8.2), 32768: (54.4, 15.1, 24.2, 6.2)}
    for n, shares in expected.items():
        total = step_bytes(m, n)
        parts = (m.layers * mlp_bytes(m), m.layers * attn_bytes(m), m.layers * kv_bytes(m, n), other_bytes(m))
        assert tuple(round(100 * p / total, 1) for p in parts) == shares
    assert round(m.layers * mlp_bytes(m) * 0.5 / 1e6) == 5436
    assert round(m.layers * kv_bytes(m, 512) * 0.7 / 1e6) == 53
    assert round(m.layers * kv_bytes(m, 32768) * 0.7 / 1e6) == 3382


def test_bounds_at_32k():
    # Table 17 byte bounds
    table = {(0.7, 1.0): 1.277, (0.6, 1.0): 1.407, (0.5, 1.0): 1.566, (1.0, 0.2): 1.217,
             (0.7, 0.2): 1.653, (0.6, 0.2): 1.877, (0.5, 0.2): 2.172}
    for (rp, rkv), b in table.items():
        assert round(bound(LLAMA8B, 32768, rp, rkv), 3) == b, (rp, rkv)


def test_matched_crossover():
    proj = {0.7: 0.180, 0.6: 0.402, 0.5: 0.995}
    sel = {0.5: -0.007, 0.3: 0.004, 0.2: 0.004, 0.1: 0.06}
    assert round(matched_crossover(LLAMA8B, proj, sel, 0.2) / K, 1) == 35.7
    assert round(matched_crossover(LLAMA8B, proj, sel, 0.5) / K, 1) == 48.1
    assert matched_crossover(LLAMA8B, proj, sel, 0.1) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

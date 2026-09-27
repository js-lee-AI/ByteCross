import json
import subprocess
import sys
from pathlib import Path

import pytest

import bytecross
from bytecross.traffic import MODELS, Model, bound

ROOT = Path(__file__).resolve().parents[1]

LLAMA_31_8B = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "vocab_size": 128256,
               "num_attention_heads": 32, "num_key_value_heads": 8, "max_position_embeddings": 131072}


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          check=True, timeout=120).stdout


def test_import_does_not_pull_torch():
    # a fresh interpreter, since pytest plugins may have imported torch already
    out = run("-c", "import sys, bytecross; print('torch' in sys.modules, 'numpy' in sys.modules)")
    assert out.split() == ["False", "False"]


def test_quickstart_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "quickstart.py"))
    assert "keep 0.5 / 0.3   n* = 74.3K tokens" in out
    assert "KV keep 0.3 at 128K   speedup bound 1.60x" in out


def test_cli():
    assert "demo" in run("-m", "bytecross", "--help")
    assert "74.3K" in run("-m", "bytecross", "demo")
    assert "n* = 65.7K" in run("-m", "bytecross", "crossover", "--model", "qwen3-8b")
    assert "--retime" in run("-m", "bytecross", "composition", "--help")


def test_from_config(tmp_path):
    assert Model.from_config(LLAMA_31_8B) == MODELS["llama-3.1-8b"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": LLAMA_31_8B}))
    assert Model.from_config(path) == MODELS["llama-3.1-8b"]
    with pytest.raises(ValueError):
        Model.from_config({**LLAMA_31_8B, "head_dim": 256})


def test_table11_bound_column():
    # byte bound of a 30% KV keep, the window column of Table 11
    bounds = [round(bound(MODELS["llama-3.1-8b"], k * 1024, rkv=0.3), 2) for k in (2, 4, 8, 16, 32, 64, 96, 128)]
    assert bounds == [1.01, 1.02, 1.05, 1.10, 1.18, 1.34, 1.48, 1.60]


def test_lazy_names_are_listed():
    assert set(bytecross._LAZY) <= set(dir(bytecross))
    with pytest.raises(AttributeError):
        bytecross.not_a_name


@pytest.mark.gpu
def test_llama3_rope_matches_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    try:
        from bytecross import harness
    except ImportError as err:
        if "TEAL not found" not in str(err):
            raise
        pytest.skip(str(err))
    scaling = harness.ROPE_SCALING["llama-3.1"]
    cfg = LlamaConfig(hidden_size=4096, num_attention_heads=32, rope_theta=500000.0,
                      rope_scaling={"rope_type": "llama3", **scaling})
    inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cpu")
    rot = harness.rope_cache(2, 128, 500000, torch.float32, scaling)
    # position 1 holds the cosine and sine of each frequency
    assert torch.allclose(torch.atan2(rot[1, :, 1], rot[1, :, 0]), inv_freq.float(), atol=1e-6)

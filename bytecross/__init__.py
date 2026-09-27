"""bytecross: where activation sparsity and KV-cache sparsity cross in LLM decoding.

    import bytecross

    m = bytecross.MODELS["llama-3.1-8b"]
    bytecross.crossover(m, rp=0.5, rkv=0.3)        # byte crossover n*, in tokens
    bytecross.bound(m, 131072, rp=0.5, rkv=0.3)    # ideal speedup of both branches at 128K

The byte account needs only model dimensions and keep ratios, so it runs anywhere.
The analyses of timing campaigns live in bytecross.campaign, .crossing, .predict,
.composition and .dispatch (numpy only). The timing and quality harness needs a GPU,
torch, flash-attn and a TEAL checkout, and is imported only when first used.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .traffic import (MODELS, Model, admissible_keep, attn_bytes, bound, crossover, crossover_batch,
                      crossover_fixed_budget, kv_bytes, matched_crossover, mlp_bytes, model_for, other_bytes,
                      proj_params, step_bytes)

# Heavy names, imported on first attribute access (PEP 562).
_LAZY = {
    "load_model": "harness",
    "sparsify_projections": "harness",
    "prefill": "harness",
    "select_and_compact": "kvselect",
}

__all__ = [
    "__version__",
    # the byte account
    "Model", "MODELS", "model_for", "crossover", "crossover_fixed_budget", "crossover_batch", "bound",
    "step_bytes", "mlp_bytes", "attn_bytes", "kv_bytes", "other_bytes", "proj_params",
    # quality-matched crossover
    "admissible_keep", "matched_crossover",
    # the harness
    "load_model", "sparsify_projections", "prefill", "select_and_compact",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)

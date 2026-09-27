# Records

These are the measurements behind the tables and figures of the paper. Every script in `experiments/` reads them through `--records`, which points here by default. Throughput is in tokens per second and step time in ms.

## Timing campaigns

One JSON file per timing cell, in the format the harness writes. The `block`, `group` and `cell` fields pair each sparse cell with the dense cell of the same block and context.

| folder | paper | rebuilt by |
|---|---|---|
| `sweep_a100/`, `sweep_a6000/` | Figure 3, Table 10 (projection, selection and composition columns), Table 2 predictions, Table 5 | `figure3_context_sweep.py`, `table2_crossings.py`, `table5_dispatch.py` |
| `sweep_3b/` | Llama-3.2-3B kernel costs in Appendix D, Table 2 prediction | `figure3_context_sweep.py`, `table2_crossings.py` |
| `keep_a100/` | Figure 6 (left), Table 2 predictions | `figure6_quality_matched.py`, `table2_crossings.py` |
| `crossing_a100_1/`, `crossing_a100_2/` | Table 2, Figure 4, Table 12 | `table2_crossings.py --full` |
| `crossing_3b/` | Table 2, Table 12 | `table2_crossings.py --full` |
| `crossing_a6000/` | Table 2, Table 13 | `table2_crossings.py --full` |
| `screen_a6000/` | Table 4, Table 16 | `table4_composition.py` |
| `retime_a6000/` | Table 4, Table 17, RTX A6000 predictions in Table 2 | `table4_composition.py`, `table2_crossings.py` |
| `retime_a100/` | Table 17, Appendix H | `table4_composition.py` |
| `sweep_a100_window/`, `sweep_a6000_window/` | Table 10 (window column) | data only |

The window cells follow the same format with mode `window`, a baseline that the harness here does not run. They have no dense cells of their own, and their dense partners are the cells of the same block and context in `sweep_a100/` and `sweep_a6000/`.

## Quality

| file or folder | paper | rebuilt by |
|---|---|---|
| `quality/grid_ppl.json` | Table 3 (perplexity), Figure 6 (middle and right), Table 4, Table 16 | `table3_retrieval.py`, `figure6_quality_matched.py`, `table4_composition.py` |
| `quality/keep_extra_ppl.json` | Figure 6 (middle and right) | `figure6_quality_matched.py` |
| `quality/long_ppl.json` | perplexity at 64K and 127K in Appendix G | `table3_retrieval.py` |
| `quality/retrieval/` | Table 3 at 32K, Table 16 (passkey) | `table3_retrieval.py`, `table4_composition.py` |
| `quality/retrieval_long/` | Table 3 at 64K and 127K, Appendix G | `table3_retrieval.py` |
| `quality/window/` | Table 3 (window row) | data only |

A perplexity record keeps the loss and token count of each of the eight windows. A retrieval record keeps every trial with its depth, the code, the model answer and whether it was correct. In `quality/window/`, `inside_window` marks placements whose needle lies among the retained sink or recent positions.

## Other measurements

| file or folder | paper | rebuilt by |
|---|---|---|
| `attention_paths_a100/` | Figure 5, Table 11 | data only |
| `projection_branch/decode_only.json` | Table 6 (decode only), Mistral-7B-v0.3 transfer in Section 4.4 | data only |
| `projection_branch/prefill_decode.json` | Table 6 (prefill and decode) | data only |
| `projection_branch/mmlu.json` | Table 7 (MMLU) | data only |
| `projection_branch/chunked_ppl.json` | Table 7 and Table 8 (perplexity) | data only |
| `projection_branch/keep_sweep.json` | Table 8 (speed) | data only |
| `batching/union_keep50.json`, `batching/union_keep70.json` | Table 14, batching in Section 4.4 | data only |
| `batching/dense_batch_*.json` | Table 15 | data only |

Each file in `attention_paths_a100/` holds the dense and window cells of one attention path, and the Table 11 speedup is the window throughput over the dense throughput at the same context. In `decode_only.json` each launch lists the throughput that the decoding loop reports together with the samples it averages, and each Llama row of Table 6 is the mean of its five launches. `keep_sweep.json` holds one launch per keep ratio. In `batching/`, `union` is the byte-weighted fraction of the projection weights that at least one sequence of a group activates, averaged over steps and groups.

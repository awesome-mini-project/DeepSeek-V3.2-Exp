## Trace analysis scripts

This folder contains post-processing scripts for the JSONL traces produced by `inference/trace.py`.

See **[docs/ANALYSIS.md](../docs/ANALYSIS.md)** for the full documentation, including:

- Design rationale and data flow
- All CLI arguments with descriptions and defaults
- Complete output JSON schema with per-field explanations
- Metrics and their relationship to KV cache placement strategies
- Worked examples for common analysis scenarios

### Quick start

```bash
# Analyze one run (uses block size from run_meta.json)
python3 analysis/analyze_trace.py --input outputs/ruler_1234567890/block64

# What-if: recompute for multiple block sizes
python3 analysis/analyze_trace.py \
  --input outputs/ruler_1234567890/block64 \
  --block-sizes 16,32,64,128

# Step-level union across all layers
python3 analysis/analyze_trace.py \
  --input outputs/ruler_1234567890/block64 \
  --step-union --key-mode request
```

### Notes

- The default trace schema does **not** store `selected_block_ids`. These scripts derive block IDs from `selected_token_pos` using your requested block size.
- For very large traces, use `--max-steps-per-stream` to sample a prefix of each request/layer stream.
- No third-party dependencies — standard library only (Python ≥ 3.8).

# DeepSeek V3.2

```shell
pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git
pip install git+https://github.com/tile-ai/tilelang
```

First convert huggingface model weights to the the format required by our inference demo. Set `MP` to match your available GPU count:
```bash
cd inference
export EXPERTS=256
export MP=8  # 根据实际 GPU 数调整 (如 4/2/1，需满足 256 % MP == 0)
export HF_CKPT_PATH=/data/models/deepseek-v3.2-exp
export SAVE_PATH=/data/models/deepseek-v3.2-exp-s
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Launch the interactive chat interface and start exploring DeepSeek's capabilities:
```bash
export CONFIG=config_671B_v3.2.json
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```

## DSA trace instrumentation

Quickstart (see [`docs/INSTRUMENTATION.md`](../docs/INSTRUMENTATION.md) for full guide):

```bash
export CKPT_PATH="${SAVE_PATH}"
./scripts/run_trace_ruler.sh
./scripts/run_trace_longbenchv2.sh
./scripts/run_trace_sharegpt.sh
```

See [`docs/INSTRUMENTATION.md`](../docs/INSTRUMENTATION.md) for:

- Where A/B/C/D instrumentation is implemented
- JSONL schema and summary outputs
- How to run RULER / LongBench v2 / ShareGPT / BurstGPT runners and scripts

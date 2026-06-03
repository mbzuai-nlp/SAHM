#!/usr/bin/env bash
# Evaluate several models on all SAHM tasks (3 seeds each).
# Secrets come from .env — never hardcode tokens here.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?set HF_TOKEN in .env}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODELS=(SILMA-9B Fanar-9B qwen2.5-7b llama-3.1-8b)
for M in "${MODELS[@]}"; do
  echo "=================  $M  ================="
  sahm-eval run --model "$M" --backend vllm --tasks all --runs 3
done
echo "done"

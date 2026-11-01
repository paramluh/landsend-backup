#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "${WORKER_GIT_UPDATE:-1}" == "1" ]]; then
  echo "Attempting to refresh worker code from git..."
  if git -C /home/xabieririzar/pose-tokenizer fetch --depth=1 origin "${WORKER_GIT_REF:-main}"; then
    git -C /home/xabieririzar/pose-tokenizer reset --hard "origin/${WORKER_GIT_REF:-main}"
    echo "Worker code refreshed to origin/${WORKER_GIT_REF:-main}"
  else
    echo "WARNING: git refresh failed; continuing with baked image code"
  fi
fi

exec uv run python tokenize_speech.py

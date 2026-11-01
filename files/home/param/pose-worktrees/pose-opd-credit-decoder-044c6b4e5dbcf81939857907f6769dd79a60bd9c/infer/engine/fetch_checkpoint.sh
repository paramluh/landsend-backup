#!/bin/bash
# Fetch the checkpoint (consolidated bf16 DCP, step 357, 67 files, 88.5 GB) from the PRIVATE HF repo InternalCan/ppm-aloe-pose-mono-8020,
# folder export-357-dcp/. Needs an HF token with access to the InternalCan org (huggingface-cli login, or HF_TOKEN).
#   bash fetch_checkpoint.sh [DEST]        -> DEST/export-357-dcp   (default DEST = ./checkpoints)
# Then: python infer.py --ckpt DEST/export-357-dcp ...
set -euo pipefail
DEST=${1:-./checkpoints}
REV=87569a04db3fe759ef4cb333e4542ca0b32755c1          # repo commit holding export-357-dcp/ (verified 2026-10-19)
mkdir -p "$DEST"
HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0} python - "$DEST" "$REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest, rev = sys.argv[1], sys.argv[2]
p = snapshot_download("InternalCan/ppm-aloe-pose-mono-8020", revision=rev, allow_patterns=["export-357-dcp/*"], local_dir=dest, max_workers=32)
print(p)
PY
test -f "$DEST/export-357-dcp/COMPLETE" && test -f "$DEST/export-357-dcp/.metadata" && echo "OK: $DEST/export-357-dcp"
if [ "${VERIFY:-0}" = 1 ]; then          # sha256 of all 67 files vs CHECKPOINT_SHA256SUMS (88.5 GB of reads)
  SUMS=$(cd "$(dirname "$0")" && pwd)/CHECKPOINT_SHA256SUMS
  (cd "$DEST" && sha256sum -c "$SUMS") && echo "SHA256 OK"
fi

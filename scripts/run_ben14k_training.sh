#!/usr/bin/env bash
# End-to-end BigEarthNet LoRA adaptation pipeline for SatQuery AI (SIH-167).
#
#   1. wait for data/BigEarthNet_14K.zip download to complete (DONE marker)
#   2. materialize per-patch folders (B02/B03/B04/B08 + labels_19.json)
#   3. fine-tune the LoRA adapters on the real BigEarthNet patches
#   4. verify the checkpoint auto-loads and merges at inference
#
# Usage: bash scripts/run_ben14k_training.sh [n_patches] [epochs]
set -uo pipefail
cd "$(dirname "$0")/.."

N_PATCHES="${1:-4000}"
EPOCHS="${2:-4}"
PY=.venv/bin/python
LOG_DIR=runs/ben14k
mkdir -p "$LOG_DIR"

echo "== [1/4] waiting for download =="
for i in $(seq 1 240); do
    if grep -q "^DONE" data/download_progress.txt 2>/dev/null; then
        echo "download complete: $(cat data/download_progress.txt)"
        break
    fi
    if ! pgrep -f "BigEarthNet_14K.zip" >/dev/null && ! ls data/BigEarthNet_14K.zip >/dev/null 2>&1; then
        echo "ERROR: downloader not running and no progress marker" >&2
        exit 1
    fi
    sleep 30
    if [ $((i % 10)) -eq 0 ]; then
        echo "  still downloading: $(ls -la data/BigEarthNet_14K.zip | awk '{print $5}') bytes"
    fi
done
if ! grep -q "^DONE" data/download_progress.txt 2>/dev/null; then
    echo "ERROR: download did not finish in time" >&2
    exit 1
fi

echo "== [2/4] materializing $N_PATCHES patches =="
$PY scripts/materialize_ben14k.py --limit "$N_PATCHES" --workers 8 \
    2>&1 | tee "$LOG_DIR/materialize.log"

echo "== [3/4] LoRA fine-tuning (epochs=$EPOCHS) =="
$PY -m satquery.training.fine_tune_bigearthnet \
    --bigearthnet-root data/BEN14K_materialized \
    --epochs "$EPOCHS" --batch-size 32 --lora-rank 8 \
    --out satquery/models/checkpoints \
    2>&1 | tee "$LOG_DIR/train.log"

echo "== [4/4] verifying checkpoint merge at inference =="
$PY - <<'EOF'
import torch
from satquery.models.single_image_vqa import RSVisionEncoder

enc = RSVisionEncoder(in_channels=4, checkpoint_dir="satquery/models/checkpoints")
print("adapter_loaded:", enc.adapter_loaded)
print("adapter_status:", enc.adapter_status)
assert enc.adapter_loaded, "checkpoint did not load"
EOF

echo "Pipeline complete. Logs in $LOG_DIR/"

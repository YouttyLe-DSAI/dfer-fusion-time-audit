#!/usr/bin/env bash
# Stage 0a-bis probes, inference + linear probes only. Reads ~/AVDFEROT read-only; writes only under results/stage0a.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
OUT=$ROOT/results/stage0a
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
mkdir -p "$OUT"
cd "$ROOT"
"$PY" -u scripts/stage0a_probe.py --src "$SRC" --run $SRC/runs/DFEW/dev/scr-8a30b61e0beb --folds 1 2 3 4 5 \
  --out "$OUT/probe_MMA_DFEW_scr.json" > "$OUT/probe_MMA_DFEW_scr.log" 2>&1 \
  && "$PY" scripts/stage0a_probe_summary.py "$OUT/probe_MMA_DFEW_scr.json" > "$OUT/probe_MMA_DFEW_scr_summary.txt" 2>&1 \
  && echo "== $(date +%H:%M) probe done" || echo "== $(date +%H:%M) probe FAILED"

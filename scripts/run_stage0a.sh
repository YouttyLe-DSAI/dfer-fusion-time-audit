#!/usr/bin/env bash
# Stage 0a, inference only. Reads code, dev checkpoints and data under ~/AVDFEROT read-only;
# writes only under $ROOT/results/stage0a. Usage: scripts/run_stage0a.sh [extra args, e.g. --limit 16 --folds 1]
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
OUT=${OUT:-$ROOT/results/stage0a}
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
ABV="host.video_ckpt=$P/pretrained/mae_face_pretrain_vit_base.pth host.audio_ckpt=$P/pretrained/audiomae_pretrained.pth data.frames_root=$P/data/{dataset}/frames data.audio_root=$P/data/{dataset}/wav"
EXTRA=("$@")
[ ${#EXTRA[@]} -eq 0 ] && EXTRA=(--folds 1 2 3 4 5)

mkdir -p "$OUT"
cd "$ROOT"
run() {
  local name=$1; shift
  "$PY" -u scripts/stage0a_spread.py --src "$SRC" --out "$OUT/$name.json" "$@" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1 \
    && echo "== $(date +%H:%M) $name done" || echo "== $(date +%H:%M) $name FAILED"
}
run MMA_DFEW_scr   --run $SRC/runs/DFEW/dev/scr-8a30b61e0beb
run MMA_MAFW_scr   --run $SRC/runs/MAFW/dev/scr-4cc4ce1d94fa
run MMA_CREMAD_scr --run $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2
run ABV_DFEW_scr   --run $P/abv_dev/runs/abvnet/DFEW/dev/scr-6e74d4117aa0 --set $ABV
echo "== $(date +%H:%M) stage0a done"

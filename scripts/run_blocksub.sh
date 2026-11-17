#!/usr/bin/env bash
# Block-subset product replacement (prereg.md, 2026-11-11), inference only. Reads ~/AVDFEROT read-only;
# writes only under $ROOT/results/blocksub.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
PY=$SRC/.venv/bin/python
ABV="host.video_ckpt=$P/pretrained/mae_face_pretrain_vit_base.pth host.audio_ckpt=$P/pretrained/audiomae_pretrained.pth data.frames_root=$P/data/{dataset}/frames data.audio_root=$P/data/{dataset}/wav"
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
log() { echo "== $(date +%H:%M) $*"; }
cd $ROOT
job() {  # name run check_prefix [--set ...]
  local name=$1 run=$2 chk=$3; shift 3
  local OUT=$ROOT/results/blocksub/$name
  mkdir -p $OUT
  for f in 1 2 3 4 5; do
    [ -s $OUT/fold$f.npz ] && { log "$name fold $f already done"; continue; }
    $PY -u scripts/blocksub_ot.py --src $SRC --run $run --fold $f --out $OUT/fold$f "$@" > $OUT/fold$f.log 2>&1 \
      && log "$name fold $f done" || log "$name fold $f FAILED"
  done
  $PY scripts/blocksub_summary.py $OUT/fold $chk > $OUT/summary.txt 2>&1 && log "$name summary done" || log "$name summary FAILED"
}
job CREMAD_mma $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2 $ROOT/results/p2_ot/CREMAD/fold
job DFEW_mma $SRC/runs/DFEW/dev/scr-8a30b61e0beb ""
job DFEW_abv $P/abv_dev/runs/abvnet/DFEW/dev/scr-6e74d4117aa0 "" --set $ABV
log "blocksub done"

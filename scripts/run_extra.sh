#!/usr/bin/env bash
# Plan dump (figures 2-3) and P6 stage 0 (prereg.md), inference only, after the E4/E5 queue.
# Reads ~/AVDFEROT read-only; writes only under $ROOT/results/{plandump,p6}.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
PY=$SRC/.venv/bin/python
ABV="host.video_ckpt=$P/pretrained/mae_face_pretrain_vit_base.pth host.audio_ckpt=$P/pretrained/audiomae_pretrained.pth data.frames_root=$P/data/{dataset}/frames data.audio_root=$P/data/{dataset}/wav"
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
log() { echo "== $(date '+%m-%d %H:%M') $*"; }
cd $ROOT
while pgrep -f "scripts/run_e3.sh|scripts/run_e45.sh" > /dev/null; do sleep 60; done
log "start"
mkdir -p results/plandump results/p6
for s in MMA_CREMAD:$SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2 MMA_DFEW:$SRC/runs/DFEW/dev/scr-8a30b61e0beb; do
  n=${s%%:*}; r=${s#*:}
  $PY -u scripts/dump_plans.py --src $SRC --run $r --out results/plandump/$n > results/plandump/$n.log 2>&1 && log "dump $n" || log "dump $n FAILED"
done
$PY -u scripts/dump_plans.py --src $SRC --run $P/abv_dev/runs/abvnet/DFEW/dev/scr-6e74d4117aa0 --set $ABV --out results/plandump/ABV_DFEW \
  > results/plandump/ABV_DFEW.log 2>&1 && log "dump ABV_DFEW" || log "dump ABV_DFEW FAILED"
for f in 1 2 3 4 5; do
  $PY -u scripts/p6_stage0.py --src $SRC --run $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2 --fold $f --out results/p6/fold$f \
    > results/p6/fold$f.log 2>&1 && log "p6 fold $f" || log "p6 fold $f FAILED"
done
log "extra done"

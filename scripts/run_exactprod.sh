#!/usr/bin/env bash
# Exact product reference (prereg.md "Exact product reference, WAR and a continuous endpoint", 2026-11-13).
# Inference only: arms base, indep_all (solver marginals) and exact_all (uniform plan a x b) on every listed checkpoint.
# Reads ~/AVDFEROT read-only; writes only under $ROOT/results/exactprod.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
PY=$SRC/.venv/bin/python
D=$SRC/runs/DFEW/dev
ABV="host.video_ckpt=$P/pretrained/mae_face_pretrain_vit_base.pth host.audio_ckpt=$P/pretrained/audiomae_pretrained.pth data.frames_root=$P/data/{dataset}/frames data.audio_root=$P/data/{dataset}/wav"
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled GIT_OPTIONAL_LOCKS=0
OUTR=$ROOT/results/exactprod
ARMS="--arms base indep_all exact_all"
log() { echo "== $(date '+%m-%d %H:%M') $*"; }
cd $ROOT
mkdir -p $OUTR/smoke
$PY -u scripts/blocksub_ot.py --src $SRC --run $D/scr-8a30b61e0beb --fold 1 --limit 2 $ARMS --out $OUTR/smoke/A0_s1_fold1 \
  > $OUTR/smoke/A0_s1_fold1.log 2>&1 && log "smoke done" || { log "smoke FAILED"; exit 1; }
job() {  # name run [--set ...]
  local name=$1 run=$2; shift 2
  local OUT=$OUTR/$name
  mkdir -p $OUT
  for f in 1 2 3 4 5; do
    [ -s $OUT/fold$f.npz ] && { log "$name fold $f already done"; continue; }
    $PY -u scripts/blocksub_ot.py --src $SRC --run $run --fold $f $ARMS --out $OUT/fold$f "$@" > $OUT/fold$f.log 2>&1 \
      && log "$name fold $f done" || log "$name fold $f FAILED"
  done
}
job A4_s1_pooled $D/pooled-fd1052e293a8
job DFEW_A0_s1 $D/scr-8a30b61e0beb
job DFEW_A0_s2 $D/scr-36d459495373
job DFEW_A0_s3 $D/scr-81feb02a6c79
job DFEW_A1 $ROOT/runs_e3/DFEW/dev/scr-e6cab78cb839
job DFEW_A2 $ROOT/runs_e6/DFEW/dev/scr-03cec607f139
job DFEW_A3 $ROOT/runs_e6/DFEW/dev/scr-30d2ed8af3a5
job DFEW_trained_eps0.02 $D/scr-a7b5c56e7d3f
job CREMAD_router $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2
job DFEW_abv $P/abv_dev/runs/abvnet/DFEW/dev/scr-6e74d4117aa0 --set $ABV
$PY scripts/exactprod_summary.py $OUTR > $OUTR/summary.log 2>&1 && log "summary done" || log "summary FAILED"
log "exactprod done"

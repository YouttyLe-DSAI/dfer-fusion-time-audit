#!/usr/bin/env bash
# Head attenuation control (prereg.md, 2026-11-13). Inference only; waits for run_exactprod.sh so that the GPU is not shared.
# Reads ~/AVDFEROT read-only; writes only under $ROOT/results/headctl.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled GIT_OPTIONAL_LOCKS=0
OUTR=$ROOT/results/headctl
ARMS="--arms base indep_10_12 ctl_meanonly ctl_frameonly ctl_rand0 ctl_rand1 ctl_rand2 ctl_perm"
log() { echo "== $(date '+%m-%d %H:%M') $*"; }
cd $ROOT
while pgrep -f run_exactprod.sh > /dev/null; do sleep 30; done
log "exactprod finished; starting"
mkdir -p $OUTR/smoke
$PY -u scripts/blocksub_ot.py --src $SRC --run $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2 --fold 1 --limit 2 $ARMS \
  --out $OUTR/smoke/CREMAD_fold1 > $OUTR/smoke/CREMAD_fold1.log 2>&1 && log "smoke done" || { log "smoke FAILED"; exit 1; }
job() {  # name run
  local name=$1 run=$2
  local OUT=$OUTR/$name
  mkdir -p $OUT
  for f in 1 2 3 4 5; do
    [ -s $OUT/fold$f.npz ] && { log "$name fold $f already done"; continue; }
    $PY -u scripts/blocksub_ot.py --src $SRC --run $run --fold $f $ARMS --out $OUT/fold$f > $OUT/fold$f.log 2>&1 \
      && log "$name fold $f done" || log "$name fold $f FAILED"
  done
}
job CREMAD_router $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2
job DFEW_A0_s1 $SRC/runs/DFEW/dev/scr-8a30b61e0beb
$PY scripts/headctl_summary.py $OUTR > $OUTR/summary.log 2>&1 && log "summary done" || log "summary FAILED"
log "headctl done"

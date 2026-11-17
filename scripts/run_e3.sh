#!/usr/bin/env bash
# E3 (prereg.md): train A1 (patch-mean cost descriptor) on DFEW dev, then evaluate it. Reads ~/AVDFEROT read-only;
# writes only under $ROOT/runs_e3 and $ROOT/results/e345.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
REF=$SRC/runs/DFEW/dev/scr-8a30b61e0beb
RUNS=$ROOT/runs_e3
OUT=$ROOT/results/e345
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled WANDB_DIR=$RUNS GIT_OPTIONAL_LOCKS=0
log() { echo "== $(date '+%m-%d %H:%M') $*"; }
mkdir -p $OUT/A1_train
cd $ROOT
for f in 1 2 3 4 5; do
  $PY -u scripts/e3_train.py --src $SRC --like $REF --fold $f --out-root $RUNS --resume > $OUT/A1_train/fold$f.log 2>&1 \
    && log "A1 train fold $f" || { log "A1 train fold $f FAILED"; exit 1; }
  if [ $f = 1 ]; then
    u=$($PY -c "import json,glob;print(json.load(open(glob.glob('$RUNS/DFEW/dev/scr-*/fold1/metrics.json')[0]))['uar'])")
    log "A1 fold 1 val10 UAR $u (A0 69.58; stop below 64.58)"
    $PY -c "import sys;sys.exit(0 if $u >= 64.58 else 1)" || { log "HEALTH CHECK: fold 1 below 64.58, stop"; exit 2; }
  fi
done
RUN=$(ls -d $RUNS/DFEW/dev/scr-*)
mkdir -p $OUT/A1
for f in 1 2 3 4 5; do
  [ -s $OUT/A1/fold$f.npz ] && continue
  $PY -u scripts/blocksub_ot.py --src $SRC --run $RUN --fold $f --out $OUT/A1/fold$f > $OUT/A1/fold$f.log 2>&1 \
    && log "A1 eval fold $f" || log "A1 eval fold $f FAILED"
done
$PY scripts/blocksub_summary.py $OUT/A1/fold > $OUT/A1/summary.txt 2>&1 && log "A1 eval summary" || log "A1 eval summary FAILED"
$PY -u scripts/e2_ot.py --src $SRC --run $RUN --split test --folds 1 2 3 4 5 --out $OUT/E2_A1_DFEW_test > $OUT/E2_A1_DFEW_test.log 2>&1 \
  && log "A1 E2 profile" || log "A1 E2 profile FAILED"
log "e3 done"

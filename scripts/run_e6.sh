#!/usr/bin/env bash
# E6 (prereg.md): train A2 (learned D7 branch) and A3 (its frozen random control) on DFEW dev, then evaluate both.
# Reads ~/AVDFEROT read-only; writes only under $ROOT/runs_e6 and $ROOT/results/e6.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
REF=$SRC/runs/DFEW/dev/scr-8a30b61e0beb
RUNS=$ROOT/runs_e6
OUT=$ROOT/results/e6
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled WANDB_DIR=$RUNS GIT_OPTIONAL_LOCKS=0
log() { echo "== $(date '+%m-%d %H:%M') $*"; }
run_of() {  # arm -> run dir
  local m
  m=$(grep -l "\"arm\": \"$1\"" $RUNS/DFEW/dev/scr-*/fold1/e6_arm.json 2>/dev/null | head -1)
  [ -n "$m" ] && dirname "$(dirname "$m")"
}
cd $ROOT
for ARM in A2 A3; do
  mkdir -p $OUT/${ARM}_train
  for f in 1 2 3 4 5; do
    $PY -u scripts/e6_train.py --src $SRC --like $REF --arm $ARM --fold $f --out-root $RUNS --resume \
      > $OUT/${ARM}_train/fold$f.log 2>&1 && log "$ARM train fold $f" || { log "$ARM train fold $f FAILED"; exit 1; }
    if [ $f = 1 ]; then
      RUN=$(run_of $ARM)
      u=$($PY -c "import json;print(json.load(open('$RUN/fold1/metrics.json'))['uar'])")
      log "$ARM fold 1 val10 UAR $u (A0 69.58; stop below 64.58)"
      $PY -c "import sys;sys.exit(0 if $u >= 64.58 else 1)" || { log "HEALTH CHECK: $ARM fold 1 below 64.58, stop"; exit 2; }
    fi
  done
done
for ARM in A2 A3; do
  RUN=$(run_of $ARM)
  mkdir -p $OUT/$ARM
  for f in 1 2 3 4 5; do
    [ -s $OUT/$ARM/fold$f.npz ] && continue
    $PY -u scripts/blocksub_ot.py --src $SRC --run $RUN --fold $f --out $OUT/$ARM/fold$f > $OUT/$ARM/fold$f.log 2>&1 \
      && log "$ARM eval fold $f" || log "$ARM eval fold $f FAILED"
  done
  $PY scripts/blocksub_summary.py $OUT/$ARM/fold > $OUT/$ARM/summary.txt 2>&1 && log "$ARM eval summary" || log "$ARM eval summary FAILED"
  $PY -u scripts/e2_ot.py --src $SRC --run $RUN --split test --folds 1 2 3 4 5 --out $OUT/E2_${ARM}_DFEW_test \
    > $OUT/E2_${ARM}_DFEW_test.log 2>&1 && log "$ARM E2 profile" || log "$ARM E2 profile FAILED"
done
$PY scripts/e6_summary.py $OUT > $OUT/summary.log 2>&1 && log "e6 summary" || log "e6 summary FAILED"
log "e6 done"

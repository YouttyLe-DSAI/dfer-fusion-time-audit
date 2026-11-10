#!/usr/bin/env bash
# P2T follow-up (b), prereg.md: order-agnostic training control for MMA-DFER (none), then the stage-0 arms on the
# new runs. Reads ~/AVDFEROT read-only; writes only under dfer-visual/runs_p2t and dfer-visual/results/p2t_train.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
RUNS=$ROOT/runs_p2t/train_shuffle
OUT=$ROOT/results/p2t_train
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled WANDB_DIR=$ROOT/runs_p2t \
       GIT_OPTIONAL_LOCKS=0
PY=$SRC/.venv/bin/python
log() { echo "== $(date '+%m-%d %H:%M') $*"; }

train() {  # name reference_run
  mkdir -p $OUT/$1
  for f in 1 2 3 4 5; do
    (cd $ROOT && $PY -u scripts/p2t_train_shuffle.py --src $SRC --like $2 --fold $f --out-root $RUNS --resume \
       > $OUT/$1/train_fold$f.log 2>&1) && log "$1 train fold $f" || { log "$1 train fold $f FAILED"; return 1; }
    if [ "$1" = dfew ] && [ $f = 1 ]; then
      u=$($PY -c "import json,glob;print(json.load(open(glob.glob('$RUNS/DFEW/confirm/none-*/fold1/metrics.json')[0]))['uar'])")
      log "dfew fold 1 control UAR $u (reference 64.88; stop below 59.88)"
      $PY -c "import sys;sys.exit(0 if $u >= 59.88 else 1)" || { log "STOP RULE: dfew fold 1 below 59.88"; return 2; }
    fi
  done
}
arms() {  # name control_run_dir
  for f in 1 2 3 4 5; do
    [ -s $OUT/$1/fold$f.npz ] && continue
    (cd $ROOT && $PY -u scripts/p2t_mma.py --src $SRC --run $2 --fold $f --out $OUT/$1/fold$f > $OUT/$1/fold$f.log 2>&1) \
      && log "$1 arms fold $f" || log "$1 arms fold $f FAILED"
  done
  $PY $ROOT/scripts/p2t_summary.py $OUT/$1 $1 > $OUT/$1/summary.txt 2>&1 && log "$1 arms summary" || log "$1 arms summary FAILED"
}

for s in ${*:-dfew cremad mafw}; do   # optional args pick the datasets, e.g. "cremad mafw" after a restart
  case $s in
    dfew) train dfew $SRC/runs/DFEW/confirm/none-c042c39441f0 || exit 1
          arms dfew $RUNS/DFEW/confirm/none-c042c39441f0 ;;
    cremad) train cremad $SRC/runs/CREMAD/dev/none-c763085c64a0 || exit 1
            arms cremad $RUNS/CREMAD/dev/none-c763085c64a0 ;;
    mafw) train mafw $SRC/runs/MAFW/confirm/none-3d586f79140b || exit 1
          arms mafw $RUNS/MAFW/confirm/none-3d586f79140b ;;
    *) log "unknown stage $s"; exit 1 ;;
  esac
done
log "p2t train control done ($*)"

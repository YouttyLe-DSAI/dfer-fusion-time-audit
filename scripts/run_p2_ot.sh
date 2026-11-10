#!/usr/bin/env bash
# P2-OT (prereg.md), inference only. Reads ~/AVDFEROT read-only; writes only under dfer-visual/results/p2_ot.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
cd $ROOT
# wait for the HiCMAE MAFW stage-0 run so the two jobs do not share the GPU
until grep -q "mafw stage0 done\|summary FAILED" p1/run_p1_mafw.log 2>/dev/null; do sleep 30; done
for spec in CREMAD:runs/CREMAD/dev/scr-f1dc1fc2dff2 MAFW:runs/MAFW/dev/scr-4cc4ce1d94fa; do
  ds=${spec%%:*}; run=$SRC/${spec#*:}
  OUT=$ROOT/results/p2_ot/$ds
  mkdir -p $OUT
  for f in 1 2 3 4 5; do
    [ -s $OUT/fold$f.npz ] && { echo "== $ds fold $f already done"; continue; }
    $PY -u scripts/p2_ot_nulls.py --src $SRC --run $run --fold $f --out $OUT/fold$f > $OUT/fold$f.log 2>&1 \
      && echo "== $(date +%H:%M) $ds fold $f done" || echo "== $(date +%H:%M) $ds fold $f FAILED"
  done
  $PY scripts/p2_ot_summary.py $OUT/fold $run > $OUT/summary.txt 2>&1 && echo "== $(date +%H:%M) $ds summary done" \
    || echo "== $(date +%H:%M) $ds summary FAILED"
done
echo "== $(date +%H:%M) p2 ot done"

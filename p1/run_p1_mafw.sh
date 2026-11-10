#!/usr/bin/env bash
# P2: HiCMAE on MAFW, stage 0 (reproduction) + P1 arms, final protocol. Writes only under p1/results/mafw.
set -u
P1=/mnt/debian135/datadeb135/dfer-visual/p1
cd $P1
export GLIBC_TUNABLES=glibc.rtld.execstack=2 PYTHONDONTWRITEBYTECODE=1
OUT=$P1/results/mafw
mkdir -p $OUT
for f in 1 2 3 4 5; do
  [ -s $OUT/fold$f.npz ] && { echo "== fold $f already done"; continue; }
  .venv/bin/python -u p1_eval.py --dataset mafw --hicmae HiCMAE --fold $f --ckpt ckpt_mafw/mafw_fold$f.pth \
    --data data/mafw --out $OUT/fold$f > $OUT/fold$f.log 2>&1 && echo "== $(date +%H:%M) fold $f done" || echo "== $(date +%H:%M) fold $f FAILED"
done
.venv/bin/python p1_summary.py $OUT/fold mafw > $OUT/summary.txt 2>&1 && echo "== $(date +%H:%M) mafw stage0 done" || echo "== $(date +%H:%M) summary FAILED"

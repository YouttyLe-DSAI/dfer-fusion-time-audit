#!/usr/bin/env bash
# P1-nulls (prereg.md), inference only. Writes only under dfer-visual/p1/{data/crema-d-trainview,results/cremad_nulls}.
set -u
P1=/mnt/debian135/datadeb135/dfer-visual/p1
cd $P1
export GLIBC_TUNABLES=glibc.rtld.execstack=2 PYTHONDONTWRITEBYTECODE=1
for f in 1 2 3 4 5; do
  mkdir -p data/crema-d-trainview/split0$f
  cp data/crema-d/split0$f/train.csv data/crema-d-trainview/split0$f/test.csv
done
OUT=$P1/results/cremad_nulls
mkdir -p $OUT
for f in 1 2 3 4 5; do
  [ -s $OUT/fold$f.npz ] && { echo "== fold $f already done"; continue; }
  .venv/bin/python -u p1_nulls.py --hicmae HiCMAE --fold $f --ckpt ckpt/cremad_fold$f.pth --data data/crema-d \
    --train-data data/crema-d-trainview --p1-npz results/cremad/fold$f.npz --out $OUT/fold$f > $OUT/fold$f.log 2>&1 \
    && echo "== $(date +%H:%M) fold $f done" || echo "== $(date +%H:%M) fold $f FAILED"
done
.venv/bin/python p1_nulls_summary.py $OUT/fold > $OUT/summary.txt 2>&1 && echo "== $(date +%H:%M) nulls done" || echo "== $(date +%H:%M) summary FAILED"

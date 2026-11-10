#!/usr/bin/env bash
# P1-desync-control (prereg.md), inference only. Writes only under dfer-visual/p1/results/cremad_desync.
set -u
P1=/mnt/debian135/datadeb135/dfer-visual/p1
cd $P1
export GLIBC_TUNABLES=glibc.rtld.execstack=2 PYTHONDONTWRITEBYTECODE=1
OUT=$P1/results/cremad_desync
mkdir -p $OUT
for f in 1 2 3 4 5; do
  [ -s $OUT/fold$f.npz ] && { echo "== fold $f already done"; continue; }
  .venv/bin/python -u p1_desync_control.py --hicmae HiCMAE --fold $f --ckpt ckpt/cremad_fold$f.pth --data data/crema-d \
    --out $OUT/fold$f > $OUT/fold$f.log 2>&1 && echo "== $(date +%H:%M) fold $f done" || echo "== $(date +%H:%M) fold $f FAILED"
done
.venv/bin/python p1_desync_summary.py $OUT/fold > $OUT/summary.txt 2>&1 && echo "== $(date +%H:%M) desync done" || echo "== $(date +%H:%M) summary FAILED"

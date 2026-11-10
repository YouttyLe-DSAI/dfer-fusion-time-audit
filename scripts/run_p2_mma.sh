#!/usr/bin/env bash
# P2-MMA (prereg.md), inference only. Reads ~/AVDFEROT read-only; writes only under dfer-visual/results/p2_mma.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
cd $ROOT
for spec in CREMAD:none:none-c763085c64a0 CREMAD:scr:scr-f1dc1fc2dff2 MAFW:none:none-07e50d760885 MAFW:scr:scr-4cc4ce1d94fa; do
  IFS=: read ds tag rid <<< "$spec"
  run=$SRC/runs/$ds/dev/$rid
  OUT=$ROOT/results/p2_mma/$ds/$tag
  mkdir -p $OUT
  for f in 1 2 3 4 5; do
    [ -s $OUT/fold$f.npz ] && { echo "== $ds $tag fold $f already done"; continue; }
    $PY -u scripts/p2_mma_nulls.py --src $SRC --run $run --fold $f --out $OUT/fold$f > $OUT/fold$f.log 2>&1 \
      && echo "== $(date +%H:%M) $ds $tag fold $f done" || echo "== $(date +%H:%M) $ds $tag fold $f FAILED"
  done
done
echo "== $(date +%H:%M) p2 mma runs done"

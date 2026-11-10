#!/usr/bin/env bash
# Stage 0b screen (prereg.md). Reads ~/AVDFEROT read-only; writes only under dfer-visual/results/s0b.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
PY=$SRC/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
cd $ROOT
until grep -q "extract done\|extract FAILED" scripts/run_s0b_extract.log 2>/dev/null; do sleep 30; done
grep -q "extract FAILED" scripts/run_s0b_extract.log && { echo "== extraction failed, stop"; exit 1; }
OUT=$ROOT/results/s0b
mkdir -p $OUT
for spec in base:b12 framewise:b12 motionpool:b12 mixer:b12 framewise:b9 motionpool:b9; do
  cfg=${spec%%:*}; blk=${spec#*:}
  [ -s $OUT/${cfg}_${blk}.npz ] && continue
  $PY -u scripts/s0b_train.py --src $SRC --cache cache/s0b --train_split $SRC/splits/DFEW_set_1_train90.txt \
    --eval_split $SRC/splits/DFEW_set_1_val10.txt --config $cfg --block $blk --out $OUT/${cfg}_${blk} \
    > $OUT/${cfg}_${blk}.log 2>&1 && echo "== $(date +%H:%M) $cfg $blk done" || echo "== $(date +%H:%M) $cfg $blk FAILED"
done
$PY scripts/s0b_summary.py $OUT > $OUT/summary.txt 2>&1 && echo "== $(date +%H:%M) s0b summary done" || echo "== s0b summary FAILED"

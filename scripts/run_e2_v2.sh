#!/usr/bin/env bash
# E2 incident fix (prereg.md, 2026-11-11): rerun HiCMAE with the D5 logits check in fp64, after run_e2.sh finishes.
# Writes only under results/e2/HiCMAE_CREMAD_test_v2 and results/e2/summary.*.
set -u
ROOT=/mnt/debian135/datadeb135/dfer-visual
OUT=$ROOT/results/e2/HiCMAE_CREMAD_test_v2
log() { echo "== $(date +%H:%M) $*"; }
until grep -q "e2 done" $ROOT/results/e2/run_e2.log 2>/dev/null; do sleep 30; done
mkdir -p $OUT
cd $ROOT/p1
for f in 1 2 3 4 5; do
  o=$OUT/fold$f
  [ -s $o.json ] && { log "v2 fold $f already done"; continue; }
  GLIBC_TUNABLES=glibc.rtld.execstack=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u e2_hicmae.py \
      --dataset cremad --hicmae HiCMAE --fold $f --ckpt ckpt/cremad_fold$f.pth --data data/crema-d-openface \
      --eval-json results/cremad_of/eval/fold$f.json --out $o > $o.log 2>&1 \
    && log "v2 fold $f done" || log "v2 fold $f FAILED"
done
cd $ROOT && PYTHONDONTWRITEBYTECODE=1 /home/debian135/AVDFEROT/AV-DFER-OT/.venv/bin/python scripts/e2_summary.py results/e2 > results/e2/summary.log 2>&1 \
  && log "summary done" || log "summary FAILED"
log "e2 v2 done"

#!/usr/bin/env bash
# P2T stage 0 (prereg.md), inference only. Reads ~/AVDFEROT and p1/ckpt* read-only; writes only under dfer-visual/results/p2t.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
P1=$ROOT/p1
OUT=$ROOT/results/p2t
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
log() { echo "== $(date +%H:%M) $*"; }
summ() { $SRC/.venv/bin/python $ROOT/scripts/p2t_summary.py $OUT/$1 $1 > $OUT/$1/summary.txt 2>&1 && log "$1 summary" || log "$1 summary FAILED"; }
hic() {  # name dataset data ckptfmt
  local R=$OUT/$1; mkdir -p $R
  for f in 1 2 3 4 5; do
    [ -s $R/fold$f.npz ] && continue
    (cd $P1 && GLIBC_TUNABLES=glibc.rtld.execstack=2 .venv/bin/python -u p2t_hicmae.py --dataset $2 --hicmae HiCMAE \
       --fold $f --ckpt $(printf "$4" $f) --data $3 --out $R/fold$f > $R/fold$f.log 2>&1) && log "$1 fold $f" || log "$1 fold $f FAILED"
  done
  summ $1
}
mma() {  # name run_dir
  local R=$OUT/$1; mkdir -p $R
  for f in 1 2 3 4 5; do
    [ -s $R/fold$f.npz ] && continue
    (cd $ROOT && $SRC/.venv/bin/python -u scripts/p2t_mma.py --src $SRC --run $2 --fold $f --out $R/fold$f > $R/fold$f.log 2>&1) \
      && log "$1 fold $f" || log "$1 fold $f FAILED"
  done
  summ $1
}
hic hicmae_cremad cremad data/crema-d-openface ckpt/cremad_fold%d.pth
mma mma_dfew   $SRC/runs/DFEW/confirm/none-c042c39441f0
mma mma_mafw   $SRC/runs/MAFW/confirm/none-3d586f79140b
mma mma_cremad $SRC/runs/CREMAD/dev/none-c763085c64a0
hic hicmae_mafw mafw data/mafw ckpt_mafw/mafw_fold%d.pth
mma ot_dfew    $SRC/runs/DFEW/confirm/scr-745f0bd422c6
mma ot_mafw    $SRC/runs/MAFW/confirm/scr-35743f9bbe71
mma ot_cremad  $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2
log "p2t stage 0 done"

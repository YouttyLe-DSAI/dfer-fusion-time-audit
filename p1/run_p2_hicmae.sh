#!/usr/bin/env bash
# P2 (prereg.md): HiCMAE under the final protocol on CREMA-D (OpenFace faces) and MAFW.
# Inference only. Writes only under dfer-visual/p1/results/{cremad_of,mafw}.
set -u
P1=/mnt/debian135/datadeb135/dfer-visual/p1
cd $P1
export GLIBC_TUNABLES=glibc.rtld.execstack=2 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
log() { echo "== $(date +%H:%M) $*"; }

job() {  # name dataset data trainview ckptfmt
  local name=$1 ds=$2 data=$3 tv=$4 ck=$5 R=$P1/results/$1
  mkdir -p $R/eval $R/nulls $R/followup $R/desync
  for f in 1 2 3 4 5; do
    c=$(printf "$ck" $f)
    [ -s $R/eval/fold$f.npz ] || { $PY -u p1_eval.py --dataset $ds --hicmae HiCMAE --fold $f --ckpt $c --data $data \
        --out $R/eval/fold$f > $R/eval/fold$f.log 2>&1 && log "$name eval $f" || log "$name eval $f FAILED"; }
  done
  $PY p1_summary.py $R/eval/fold $ds > $R/eval/summary.txt 2>&1; log "$name stage0: $(grep 'stage 0 verdict' $R/eval/summary.txt)"
  for f in 1 2 3 4 5; do
    c=$(printf "$ck" $f)
    [ -s $R/nulls/fold$f.npz ] || { $PY -u p1_nulls.py --dataset $ds --hicmae HiCMAE --fold $f --ckpt $c --data $data \
        --train-data $tv --out $R/nulls/fold$f > $R/nulls/fold$f.log 2>&1 && log "$name nulls $f" || log "$name nulls $f FAILED"; }
  done
  $PY p1_nulls_summary.py $R/nulls/fold > $R/nulls/summary.txt 2>&1; log "$name nulls summary"
  for f in 1 2 3 4 5; do
    c=$(printf "$ck" $f)
    [ -s $R/followup/fold$f.npz ] || { $PY -u p1_followup.py --dataset $ds --hicmae HiCMAE --fold $f --ckpt $c --data $data \
        --out $R/followup/fold$f > $R/followup/fold$f.log 2>&1 && log "$name followup $f" || log "$name followup $f FAILED"; }
  done
  $PY p1_followup_summary.py $R/followup/fold > $R/followup/summary.txt 2>&1; log "$name followup summary"
  for f in 1 2 3 4 5; do
    c=$(printf "$ck" $f)
    [ -s $R/desync/fold$f.npz ] || { $PY -u p1_desync_control.py --dataset $ds --hicmae HiCMAE --fold $f --ckpt $c --data $data \
        --out $R/desync/fold$f > $R/desync/fold$f.log 2>&1 && log "$name desync $f" || log "$name desync $f FAILED"; }
  done
  $PY p1_desync_summary.py $R/desync/fold > $R/desync/summary.txt 2>&1; log "$name desync summary"
}

until grep -q "== .* done$" ../openface/extract.log 2>/dev/null; do sleep 30; done
log "faces: $(ls ../data/CREMA-D/face_aligned | wc -l) aligned dirs; failures: $(grep -c FAILED ../openface/extract.log)"
job cremad_of cremad data/crema-d-openface data/crema-d-openface-trainview ckpt/cremad_fold%d.pth
# MAFW stage 0 already ran under the final protocol in results/mafw (fold*.npz); reuse it as eval
mkdir -p results/mafw/eval && for f in 1 2 3 4 5; do for e in json npz log; do [ -e results/mafw/eval/fold$f.$e ] || cp results/mafw/fold$f.$e results/mafw/eval/; done; done
job mafw mafw data/mafw data/mafw-trainview ckpt_mafw/mafw_fold%d.pth
log "p2 hicmae done"

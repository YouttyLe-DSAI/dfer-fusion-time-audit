#!/usr/bin/env bash
# E2 (prereg.md), inference only. Reads code, checkpoints and data read-only (~/AVDFEROT, p1/ckpt);
# writes only under $ROOT/results/e2. Usage: scripts/run_e2.sh
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
OUT=$ROOT/results/e2
PY=$SRC/.venv/bin/python
ABV="host.video_ckpt=$P/pretrained/mae_face_pretrain_vit_base.pth host.audio_ckpt=$P/pretrained/audiomae_pretrained.pth data.frames_root=$P/data/{dataset}/frames data.audio_root=$P/data/{dataset}/wav"
log() { echo "== $(date +%H:%M) $*"; }
mkdir -p "$OUT"
cd "$ROOT"

ot() {
  local name=$1; shift
  [ -s "$OUT/$name.json" ] && { log "$name already done"; return; }
  PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled \
    "$PY" -u scripts/e2_ot.py --src "$SRC" --split test --folds 1 2 3 4 5 --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1 \
    && log "$name done" || log "$name FAILED"
}

# primary
ot OT_CREMAD_test --run $SRC/runs/CREMAD/dev/scr-f1dc1fc2dff2
mkdir -p "$OUT/HiCMAE_CREMAD_test"
for f in 1 2 3 4 5; do
  o=$OUT/HiCMAE_CREMAD_test/fold$f
  [ -s $o.json ] && { log "hicmae fold $f already done"; continue; }
  (cd p1 && GLIBC_TUNABLES=glibc.rtld.execstack=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u e2_hicmae.py \
      --dataset cremad --hicmae HiCMAE --fold $f --ckpt ckpt/cremad_fold$f.pth --data data/crema-d-openface \
      --eval-json results/cremad_of/eval/fold$f.json --out $o > $o.log 2>&1) \
    && log "hicmae fold $f done" || log "hicmae fold $f FAILED"
done
# secondary
ot OT_DFEW_test --run $SRC/runs/DFEW/dev/scr-8a30b61e0beb
ot ABV_DFEW_test --run $P/abv_dev/runs/abvnet/DFEW/dev/scr-6e74d4117aa0 --set $ABV
log "e2 done"

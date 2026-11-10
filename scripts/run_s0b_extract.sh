#!/usr/bin/env bash
# Stage 0b feature cache (prereg.md). Reads ~/AVDFEROT read-only; writes only under dfer-visual/cache/s0b.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
export PYTHONDONTWRITEBYTECODE=1
cd $ROOT
$SRC/.venv/bin/python -u scripts/s0b_extract.py --src $SRC --frames_root "/mnt/debian135/datadeb135/AVDFEROT/data/{dataset}/frames" \
  --anns $SRC/annotation/DFEW_set_1_train.txt $SRC/annotation/DFEW_set_1_test.txt \
  --video_ckpt $P/pretrained/mae_face_pretrain_vit_base.pth --out cache/s0b > cache/s0b_extract.log 2>&1 \
  && echo "== $(date +%H:%M) extract done" || echo "== $(date +%H:%M) extract FAILED"

#!/usr/bin/env bash
# HiCMAE's published environment (README): Python 3.8, torch 1.10.1+cu113, timm 0.4.12, ...
# Everything (python, uv cache, venv) stays under dfer-visual/p1.
set -eu
P1=/mnt/debian135/datadeb135/dfer-visual/p1
export UV_CACHE_DIR=$P1/.uv-cache UV_PYTHON_INSTALL_DIR=$P1/.uv-python
UV=$HOME/.local/bin/uv
cd $P1
[ -d .venv ] || $UV venv -p 3.8 .venv
$UV pip install -p .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu113 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113
$UV pip install -p .venv/bin/python \
  timm==0.4.12 einops==0.6.1 decord==0.6.0 opencv-python==4.7.0.72 numpy==1.23.5 scipy==1.10.0 \
  pandas==1.5.3 scikit-learn==1.2.1 tensorboardX==2.6.1 soundfile==0.12.1 Pillow==9.5.0
.venv/bin/python -c "import torch, torchaudio, timm, decord, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('timm', timm.__version__)"
$UV pip freeze -p .venv/bin/python > requirements-lock.txt
echo "== env done"

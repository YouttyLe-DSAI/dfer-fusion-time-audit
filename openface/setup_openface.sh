#!/usr/bin/env bash
# Build OpenFace 2.2.0 (FeatureExtraction) without sudo. Everything stays under dfer-visual/.
#   tools/bin/micromamba, tools/mamba (root prefix, pkgs cache), openface/OpenFace (source + build)
set -euo pipefail
D=/mnt/debian135/datadeb135/dfer-visual
export MAMBA_ROOT_PREFIX=$D/tools/mamba CONDA_PKGS_DIRS=$D/tools/mamba/pkgs XDG_CACHE_HOME=$D/tools/cache
MM=$D/tools/bin/micromamba
ENV=$MAMBA_ROOT_PREFIX/envs/openface
log() { echo "== $(date +%H:%M:%S) $*"; }

if [ ! -x $MM ]; then
  log "micromamba"
  curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $D/tools bin/micromamba
fi
$MM --version

if [ ! -d $ENV ]; then
  log "env"
  $MM create -y -n openface -c conda-forge --override-channels \
    cmake make pkg-config cxx-compiler libopencv "libboost-devel=1.82" openblas "dlib>=19.13" ffmpeg
fi

cd $D/openface
if [ ! -d OpenFace ]; then
  log "clone OpenFace 2.2.0"
  git clone -q --depth 1 --branch OpenFace_2.2.0 https://github.com/TadasBaltrusaitis/OpenFace.git
fi
cd OpenFace
log "models"
bash download_models.sh || true
ls -la lib/local/LandmarkDetector/model/patch_experts/ | head -20

log "build"
mkdir -p build && cd build
$MM run -n openface cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_PREFIX_PATH=$ENV \
  -D CMAKE_INSTALL_RPATH=$ENV/lib -D CMAKE_BUILD_WITH_INSTALL_RPATH=ON .. 2>&1 | tail -25
$MM run -n openface make -j12 FeatureExtraction 2>&1 | tail -40
ls -la bin/
log "done"

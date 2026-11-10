#!/usr/bin/env bash
# Configure + build OpenFace 2.2.0 FeatureExtraction in the micromamba env (after setup_openface.sh made the env).
set -uo pipefail
D=/mnt/debian135/datadeb135/dfer-visual
export MAMBA_ROOT_PREFIX=$D/tools/mamba CONDA_PKGS_DIRS=$D/tools/mamba/pkgs XDG_CACHE_HOME=$D/tools/cache
MM=$D/tools/bin/micromamba
E=$MAMBA_ROOT_PREFIX/envs/openface
cd $D/openface/OpenFace
rm -rf build && mkdir -p build && cd build
echo "== $(date +%H:%M:%S) configure"
$MM run -n openface cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_PREFIX_PATH=$E \
  -D OpenBLAS_LIB=$E/lib/libopenblas.so -D OpenBLAS_INCLUDE_DIR=$E/include \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 -D dlib_DIR=$D/tools/dlib/lib/cmake/dlib \
  -D CMAKE_INSTALL_RPATH=$E/lib -D CMAKE_BUILD_WITH_INSTALL_RPATH=ON .. 2>&1 | grep -v "^--   " | tail -30
echo "== $(date +%H:%M:%S) make"
$MM run -n openface make -j12 FeatureExtraction > make.log 2>&1; echo "make exit $?"; grep -E " error|Error |\*\*\*" make.log | head -40; grep "Built target" make.log
ls -la bin/ 2>/dev/null
echo "== $(date +%H:%M:%S) build finished"

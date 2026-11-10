#!/usr/bin/env bash
# dlib C++ (conda-forge ships only the Python module) -> tools/dlib, then rebuild OpenFace against it.
set -uo pipefail
D=/mnt/debian135/datadeb135/dfer-visual
export MAMBA_ROOT_PREFIX=$D/tools/mamba CONDA_PKGS_DIRS=$D/tools/mamba/pkgs XDG_CACHE_HOME=$D/tools/cache
MM=$D/tools/bin/micromamba
E=$MAMBA_ROOT_PREFIX/envs/openface
cd $D/openface
[ -d dlib-src ] || git clone -q --depth 1 --branch v19.24.6 https://github.com/davisking/dlib.git dlib-src
rm -rf dlib-build && mkdir dlib-build && cd dlib-build
echo "== $(date +%H:%M:%S) dlib configure"
$MM run -n openface cmake ../dlib-src -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$D/tools/dlib \
  -D CMAKE_PREFIX_PATH=$E -D DLIB_NO_GUI_SUPPORT=ON -D DLIB_USE_CUDA=OFF -D DLIB_USE_BLAS=ON -D DLIB_USE_LAPACK=OFF \
  -D DLIB_PNG_SUPPORT=ON -D DLIB_JPEG_SUPPORT=ON -D BUILD_SHARED_LIBS=OFF -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 2>&1 | grep -v "^--   " | tail -12
echo "== $(date +%H:%M:%S) dlib build"
$MM run -n openface make -j12 install > make_install.log 2>&1; echo "make install exit $?"; grep -E " error|Error |\*\*\*" make_install.log | head -20
ls $D/tools/dlib/lib/cmake/dlib
echo "== $(date +%H:%M:%S) openface"
grep -n dlib_DIR $D/openface/build_openface.sh
bash $D/openface/build_openface.sh

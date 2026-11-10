#!/usr/bin/env bash
# CREMA-D faces with OpenFace 2.2.0, as HiCMAE's preprocess/face_and_audio/extract_face.py (__main__):
#   FeatureExtraction -f <clip>.flv -out_dir openface/<clip> -simalign -simsize 256 -format_aligned jpg -nomask
# then face_aligned/<clip> -> openface/<clip>/<clip>_aligned (the author copies; a symlink is equivalent).
# Then HiCMAE-format split files with the frames path remapped (audio: the wav used in P1).
set -uo pipefail
D=/mnt/debian135/datadeb135/dfer-visual
OF=$D/openface/OpenFace/build/bin/FeatureExtraction
IN=$D/data/CREMA-D/VideoFlash
OUT=$D/data/CREMA-D/openface
AL=$D/data/CREMA-D/face_aligned
mkdir -p $OUT $AL
N=${N:-0}   # N>0: only the first N videos (smoke test)
list() { if [ "$N" -gt 0 ]; then ls $IN/*.flv | head -n $N; else ls $IN/*.flv; fi; }
echo "== $(date +%H:%M:%S) extract ($(list | wc -l) videos)"
list | xargs -P ${P:-16} -I{} bash -c '
  n=$(basename "{}" .flv)
  [ -d "'$OUT'/$n/${n}_aligned" ] && [ -s "'$OUT'/$n/$n.csv" -o -n "$(ls -A "'$OUT'/$n/${n}_aligned" 2>/dev/null)" ] && exit 0
  "'$OF'" -f "{}" -out_dir "'$OUT'/$n" -simalign -simsize 256 -format_aligned jpg -nomask > "'$OUT'/$n.log" 2>&1 || echo "FAILED $n"'
echo "== $(date +%H:%M:%S) link"
for d in $OUT/*/; do n=$(basename $d); [ -e $AL/$n ] || ln -s $d${n}_aligned $AL/$n; done
echo "aligned dirs: $(ls $AL | wc -l)"
echo "== $(date +%H:%M:%S) done"

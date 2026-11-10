#!/usr/bin/env bash
# Download HiCMAE's released CREMA-D fine-tuned checkpoints and eval logs (README links, MIT licence).
# SharePoint "anyone with the link" shares need a session cookie first, then &download=1.
set -u
cd /mnt/debian135/datadeb135/dfer-visual/p1/ckpt
B="https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi"
declare -A LOG=(
  [1]="EXf8jT_e7uFIpUIiuQISAGEBDuMdeZXM2L7pTN7VbDA_DA?e=QAPZ0n"
  [2]="EbU6et8b9DpBpyWI26lDhjsB2JdDwRt8yHOA0k5JNX0L-A?e=nnkPp6"
  [3]="EelLU_bTRsZDkoNXDbw1q18BI9Yes-3TKt9zcCwqGJPsQg?e=QyymZS"
  [4]="EY8mercoIS9Koa18BiKlOesBGco7EkKW8It7y-3mUkWyeA?e=8Y2B0Q"
  [5]="EU5dZKG1mHVHpN_W6-j9cwcBYk89-51rzGRGLkt4vMp3CA?e=sflC8v"
)
declare -A CKPT=(
  [1]="ERtJlWmrrk5LuKycdrmlXFYBbGFdmvjI6oNvLf-6OCYZWA?e=j7YXkP"
  [2]="Ef_tG1Eh7nNOj5BV3ljvKMcBDWkzt17ytaUknSPgg28jBQ?e=dqTvez"
  [3]="EUIvB5E1029Atp0kmIaNsOEBIjFItoyI_B8WCCWxNCTRyw?e=aT7Ha0"
  [4]="EelA-2vHrOdBj2VFX5xWBwgBXGeHrLCKRCwC-ToLkQqVPA?e=UqhN17"
  [5]="EaS5Lz94qLdErJPkXnIaHUUBO24c56cuaMIcMRRxppPAdw?e=TJ5ysA"
)
get() {  # $1 share id, $2 output file
  local s="$B/$1"
  curl -sSL -c jar.txt -o /dev/null "$s" && \
  curl -sSL -b jar.txt -o "$2.part" -w "$2: http %{http_code} %{size_download} bytes\n" "$s&download=1" && mv "$2.part" "$2"
}
for f in 1 2 3 4 5; do
  [ -s "cremad_fold${f}_log.txt" ] || get "${LOG[$f]}" "cremad_fold${f}_log.txt"
  [ -s "cremad_fold${f}.pth" ] || get "${CKPT[$f]}" "cremad_fold${f}.pth"
done
rm -f jar.txt fold1_log.txt
ls -la
sha256sum cremad_fold*.pth > SHA256SUMS
echo "== download done"

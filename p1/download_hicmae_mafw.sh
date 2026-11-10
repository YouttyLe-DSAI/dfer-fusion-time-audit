#!/usr/bin/env bash
# Download HiCMAE's released MAFW fine-tuned checkpoints and logs (README links, MIT licence).
# SharePoint "anyone with the link" shares need a session cookie first, then &download=1.
set -u
mkdir -p /mnt/debian135/datadeb135/dfer-visual/p1/ckpt_mafw
cd /mnt/debian135/datadeb135/dfer-visual/p1/ckpt_mafw
B="https://unioulu-my.sharepoint.com/:u:/g/personal/lsun24_univ_yo_oulu_fi"
declare -A LOG=(
  [1]="EXPu0Nlku05LkOyXD7UrgKIBFEWMKpXee9JkQ9Tp3597kw?e=uDV3l0"
  [2]="EcKoXaBqwG9IhRukKstH_4IBc8dpIWWl-fQQSiRx2B0OdA?e=XF2DhJ"
  [3]="Ec1DpC4iekxHsbn3Ixf1-QMB1wcTRv7R-Ba7XpKvtg5idg?e=hj5Od9"
  [4]="EbA-glMilZBDmE3oytwHdWoBWF9yUD9-rMzyiwosnDbnwg?e=eOji11"
  [5]="EVuvCgeTE9xFlFP21ZCAFaYBQRS7LSeXKVCvLG6MT77NuA?e=hIkdLe"
)
declare -A CKPT=(
  [1]="ES-eTQiMgSZKno9KyUZy1cgB5irFyRkGDIoLsChy7G9VmQ?e=li2MbK"
  [2]="EVESyhEmuwJKpswfhfkWtsIBHVjsIoIVgi9BIP1LRl_KSw?e=fSwwWk"
  [3]="ER74xfvkeFFGtfPzPH9CHhAB_qtuDI_rdKKKN-f4ep05ZA?e=5Gz7Iw"
  [4]="EXgUG2jtGopElcHJKA3bSokB4p6gv1ds-CfXhE2AaAdEuQ?e=5YTm4T"
  [5]="EWpqPBE74IlIooG0JoOPcrABjxsS3iIsLiDeEbLJnllidA?e=EhoobV"
)
get() {  # $1 share id, $2 output file
  local s="$B/$1"
  curl -sSL -c jar.txt -o /dev/null "$s" && \
  curl -sSL -b jar.txt -o "$2.part" -w "$2: http %{http_code} %{size_download} bytes\n" "$s&download=1" && mv "$2.part" "$2"
}
for f in 1 2 3 4 5; do
  [ -s "mafw_fold${f}_log.txt" ] || get "${LOG[$f]}" "mafw_fold${f}_log.txt"
  [ -s "mafw_fold${f}.pth" ] || get "${CKPT[$f]}" "mafw_fold${f}.pth"
done
rm -f jar.txt
ls -la
sha256sum mafw_fold*.pth > SHA256SUMS
echo "== download done"

#!/usr/bin/env bash
# E4 (eps sweep on A0) and E5 (A4 = trained pooled runs), plus A0 seed replicates (prereg.md). Inference only.
# Reads ~/AVDFEROT read-only; writes only under $ROOT/results/e345.
set -u
P=/home/debian135/AVDFEROT
SRC=$P/AV-DFER-OT
ROOT=/mnt/debian135/datadeb135/dfer-visual
OUT=$ROOT/results/e345
PY=$SRC/.venv/bin/python
D=$SRC/runs/DFEW/dev
export PYTHONDONTWRITEBYTECODE=1 AVDFER_PRETRAINED=$P/pretrained WANDB_MODE=disabled
log() { echo "== $(date '+%m-%d %H:%M') $*"; }
cd $ROOT
# run after E3 (training alone is ~8x faster than sharing CPU and GPU with this queue)
while pgrep -f "scripts/run_e3.sh|scripts/run_extra.sh" > /dev/null; do sleep 60; done
log "start (E3 queue finished)"
job() {  # outdir run [blocksub args...]
  local name=$1 o=$OUT/$1 run=$2; shift 2
  mkdir -p $o
  for f in 1 2 3 4 5; do
    [ -s $o/fold$f.npz ] && continue
    $PY -u scripts/blocksub_ot.py --src $SRC --run $run --fold $f --arms base indep_all --out $o/fold$f "$@" > $o/fold$f.log 2>&1 \
      || { log "$name fold $f FAILED"; return 1; }
  done
  $PY scripts/blocksub_summary.py $o/fold > $o/summary.txt 2>&1 && log "$name done" || log "$name summary FAILED"
}
resid_high() {  # dir -> exit 0 if the fold-mean row or col residual > 0.01
  $PY -c "
import json,sys
r=[json.load(open('$1/fold%d.json'%f))['plan_residual'] for f in range(1,6)]
m=max(sum(x['row_mean'] for x in r)/5, sum(x['col_mean'] for x in r)/5)
print('mean residual', m); sys.exit(0 if m > 0.01 else 1)"
}
# E4: eps sweep on A0 seed 1, trained n_iters 10; n_iters 100 where the residual exceeds 0.01
for e in 0.01 0.02 0.05 0.1 0.2; do
  job eps/A0_eps$e $D/scr-8a30b61e0beb --set fusion.eps=$e
  if resid_high $OUT/eps/A0_eps$e; then job eps/A0_eps${e}_it100 $D/scr-8a30b61e0beb --set fusion.eps=$e fusion.n_iters=100; fi
done
job eps/trained_eps0.02 $D/scr-a7b5c56e7d3f
# run-to-run reference: A0 seeds 2, 3 (seed 1 = eps/A0_eps0.05)
job seeds/A0_s2 $D/scr-36d459495373
job seeds/A0_s3 $D/scr-81feb02a6c79
# E5: A4 = trained pooled runs, native and with the SCR plan switched on at inference
SCR="fusion.rule=scr fusion.eps=0.05 fusion.n_iters=10 fusion.tau=1000000.0 fusion.video_token=cls fusion.mask_padding=true"
for s in 1:pooled-fd1052e293a8 2:pooled-f3e0b427d7cc 3:pooled-4c1083790ffa; do
  n=${s%%:*}; r=$D/${s#*:}
  job A4/s${n}_native $r
  job A4/s${n}_scr $r --set $SCR
done
log "e45 done"

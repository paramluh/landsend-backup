#!/bin/bash
#SBATCH -J textB-restore-demos -p all -w slinky-15 -N1 -n1 -c 1 --mem=1G --gres=none -t 06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB/logs/restore-%j.log
# Restores EXACTLY the demos / servers paused at 10-20 23:43 PT for the Aloe text-B FT (ppm/DEMOS_PAUSED_1020.txt, block "10-20 23:30 PT"), once
# the FT job has left slinky-7/15/18 (submit with --dependency=afterany:<FT job>). The 9 sbatch lines are the stopped jobs' sacct SubmitLines
# (minus --parsable; asr16k without its old staging dependency, its node-local export-253 is checked first). One job at a time: each must reach
# RUNNING on the expected GPUs before the next; then every URL file must change to a NEW quick-tunnel URL that answers HTTP 200, and the Aloe
# demos must log "MODEL ready". Results -> logs/restore-<job>.log + RESTORED.json (the coordinator records them in DEMOS_PAUSED / RESERVED_GPUS).
set -u
RJ=${SLURM_JOB_ID:-x}
# submit the demos with a clean Slurm environment (as from the login node): drop this job's SLURM_* / SBATCH_* / SRUN_* variables
for v in $(compgen -e | grep -E '^(SLURM_|SBATCH_|SRUN_|SALLOC_)'); do unset "$v"; done
N=/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB; D=/data/home/amartyavarma/ppm/aloe-m1vox-live; K=$D/demo11k
W=/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/export-253; O=/data/home/amartyavarma/ppm/aloe-v2vox-raw/export-253
X=/scratch/amartyavarma/aloe-v2vox-asr-w15973/exports/export-253
SAM=/data/home/amartyavarma/sam-pose-server; PPS=/data/home/amartyavarma/ppm/ppm-pose-server
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
FT=${FT_JOB:-}
[ -z "$FT" ] || { s=$(squeue -h -j $FT -o %T 2>/dev/null); [ -z "$s" ] || { echo "[$(pt)] ABORT: FT job $FT still $s"; exit 1; }; }
echo "[$(pt)] restore job $RJ: FT ${FT:-?} gone; restoring 9 jobs"
RG=/data/home/amartyavarma/ppm/RESERVED_GPUS.txt; PYR=/data/home/amartyavarma/venvs/qwen-s2/bin/python
cp -a $RG $RG.bak-restore-$RJ
# the FT's GPU lines must go first: sam-pose-server's start.sbatch skips every GPU RESERVED_GPUS lists for another owner
$PYR -I - $RG "$(pt)" "${FT:-?}" <<'EOF'
import re, sys
p, t, ft = sys.argv[1:4]
L = open(p).read().split("\n")
out = [("# DONE " + t + " (text-B FT job " + ft + " ended): " + ln) if re.match(r"^slinky-(7|15|18) \d+ aloe-textB-ft", ln) else ln for ln in L]
open(p, "w").write("\n".join(out))
EOF
echo "[$(pt)] RESERVED_GPUS: aloe-textB-ft lines retired ($(grep -c 'aloe-textB-ft' $RG) mentions, $(grep -cE '^slinky-(7|15|18) [0-9]+ aloe-textB-ft' $RG) active)"
declare -A OLD
for f in $D/URL.txt $D/URL2.txt $D/URL3.txt $D/URL_asr16k.txt $D/URL_11k.txt $K/URL_11k2.txt $K/URL_11k3.txt $SAM/URL $PPS/URL; do OLD[$f]=$(cat $f 2>/dev/null); done
RES=()
# sub <label> <node> <expected IDX> <URL file> <out log pattern or -> sbatch args...
sub() {
  local lab=$1 node=$2 idx=$3 uf=$4 olog=$5; shift 5
  local j st got i
  j=$(sbatch --parsable "$@") || { echo "[$(pt)] $lab SUBMIT FAILED: sbatch $*"; RES+=("$lab|FAIL-submit|||"); return 1; }
  for i in $(seq ${WAITN:-120}); do st=$(squeue -h -j $j -o %T); [ "$st" = RUNNING ] && break; sleep 5; done
  got=$(scontrol show job -d $j | grep -oE "gpu[^ ]*IDX:[0-9,-]+" | grep -oE "IDX:[0-9,-]+" | head -1)
  echo "[$(pt)] $lab job $j state $st node $(squeue -h -j $j -o %N) GPUs $got (expect IDX:$idx)"
  [ "$got" = "IDX:$idx" ] || echo "[$(pt)] !!! $lab got $got, expected IDX:$idx"
  RES+=("$lab|$j|$node|$got|$uf|${olog/\%j/$j}")
}
# ---- slinky-7: new-FT Aloe (0-1 / 2-3 / 4-5), then asr16k (6-7)
sub m1vox-live slinky-7 0-1 $D/URL.txt $D/logs/demo-%j.out -J m1vox-live -w slinky-7 --export=ALL,CK=$W,PPMLIVE_SWAP_TO=$W $D/demo.sbatch
sub m1vox-live2 slinky-7 2-3 $D/URL2.txt $D/logs/demo2-%j.out -J m1vox-live2 -w slinky-7 --mem=100G -o $D/logs/demo2-%j.out --export=ALL,FRONT=8795,BACK=8796,PPMLIVE_STATE_DIR=$D/inst2,URLF=$D/URL2.txt,CK=$W,PPMLIVE_SWAP_TO=$W $D/demo.sbatch
sub m1vox-live3 slinky-7 4-5 $D/URL3.txt $D/logs/demo3-%j.out -J m1vox-live3 -w slinky-7 --mem=100G -o $D/logs/demo3-%j.out --export=ALL,FRONT=8815,BACK=8816,PPMLIVE_STATE_DIR=$D/inst3,URLF=$D/URL3.txt,CK=$W,PPMLIVE_SWAP_TO=$W $D/demo.sbatch
# ---- slinky-18: 11k demo (0-1 / 2-3 / 4-5, as stopped)
sub m1vox-11k slinky-18 0-1 $D/URL_11k.txt $D/logs/demo11k-%j.out -J m1vox-11k -w slinky-18 -c 16 --mem=100G -o $D/logs/demo11k-%j.out --export=ALL,FRONT=8825,BACK=8826,PPMLIVE_STATE_DIR=$K/inst1,URLF=$D/URL_11k.txt,CK=$O,PPMLIVE_SWAP_TO=$O $D/demo.sbatch
sub m1vox-11k2 slinky-18 2-3 $K/URL_11k2.txt $D/logs/demo11k2-%j.out -J m1vox-11k2 -w slinky-18 -c 14 --mem=100G -o $D/logs/demo11k2-%j.out --export=ALL,FRONT=8835,BACK=8836,PPMLIVE_STATE_DIR=$K/inst2,URLF=$K/URL_11k2.txt,CK=$O,PPMLIVE_SWAP_TO=$O $D/demo.sbatch
sub m1vox-11k3 slinky-18 4-5 $K/URL_11k3.txt $D/logs/demo11k3-%j.out -J m1vox-11k3 -w slinky-18 -c 14 --mem=100G -o $D/logs/demo11k3-%j.out --export=ALL,FRONT=8845,BACK=8846,PPMLIVE_STATE_DIR=$K/inst3,URLF=$K/URL_11k3.txt,CK=$O,PPMLIVE_SWAP_TO=$O $D/demo.sbatch
# ---- slinky-15: SAM first (0-3), then PPM pose (4-7)
sub sam-pose-server slinky-15 0-3 $SAM/URL - $SAM/code/start.sbatch
sub ppm-pose-server slinky-15 4-7 $PPS/URL - $PPS/code/start.sbatch
# ---- slinky-7 asr16k LAST: the FT's HF backup job (afterok, 4 CPUs / node) holds slinky-7 CPUs and slinky-7's four demos need all 64, so
#      asr16k (-c 16) starts when the backup ends (Slurm order: backup submitted first). Submitted without waiting; the verify loop waits
#      up to 3.5 h for its new URL and re-reads every job's GPUs at the end.
J1=$(echo "${RES[0]}" | cut -d'|' -f2)
if srun --overlap --jobid=$J1 -w slinky-7 -N1 -n1 -c1 --mem=256M --cpu-bind=none test -f $X/COMPLETE; then
  echo "[$(pt)] asr16k weights $X/COMPLETE present on slinky-7"
  WAITN=2 sub m1vox-asr16k slinky-7 6-7 $D/URL_asr16k.txt $D/logs/demoasr16k-%j.out -J m1vox-asr16k -w slinky-7 -c 16 --mem=100G -o $D/logs/demoasr16k-%j.out --export=ALL,FRONT=8865,BACK=8866,PPMLIVE_STATE_DIR=$D/asr16k,URLF=$D/URL_asr16k.txt,CK=$X,PPMLIVE_SWAP_TO=$X $D/demo.sbatch
else
  echo "[$(pt)] !!! asr16k: $X/COMPLETE MISSING on slinky-7: staging first"
  SJ=$(sbatch --parsable $D/asr16k/stage_weights.sbatch)
  WAITN=2 sub m1vox-asr16k slinky-7 6-7 $D/URL_asr16k.txt $D/logs/demoasr16k-%j.out --dependency=afterok:$SJ -J m1vox-asr16k -w slinky-7 -c 16 --mem=100G -o $D/logs/demoasr16k-%j.out --export=ALL,FRONT=8865,BACK=8866,PPMLIVE_STATE_DIR=$D/asr16k,URLF=$D/URL_asr16k.txt,CK=$X,PPMLIVE_SWAP_TO=$X $D/demo.sbatch
fi
# ---- RESERVED_GPUS: re-add the paused slinky-7 / slinky-18 lines with the new job ids (the pose servers append their own on start)
jid() { for r in "${RES[@]}"; do [ "${r%%|*}" = "$1" ] && { echo "$r" | cut -d'|' -f2; return; }; done; echo "?"; }
{ echo "# $(pt) (restore job $RJ): demos restored after the Aloe text-B FT (see ppm/DEMOS_PAUSED_1020.txt, block 10-20 23:30 PT)"
  echo "slinky-18 4 chat-cli   # 10-19 coordinator: the terminal chat CLIs (bin/chat-intl6242, bin/chat-ppm12829 -> ppm/chatcli/chatlaunch); the tmap temperature sweep frees it (restored; GPU 4 is also in m1vox-11k3's pair, as before the pause)"
  for g in 0 1; do echo "slinky-7 $g aloe-demo-new   # m1vox-live $(jid m1vox-live): Aloe w-14503-init FT demo, instance 1 (+ front :8785 / LB / tunnel, URL.txt)"; done
  for g in 2 3; do echo "slinky-7 $g aloe-demo-new   # m1vox-live2 $(jid m1vox-live2): instance 2"; done
  for g in 4 5; do echo "slinky-7 $g aloe-demo-new   # m1vox-live3 $(jid m1vox-live3): instance 3"; done
  for g in 6 7; do echo "slinky-7 $g aloe-demo-asr16k   # m1vox-asr16k $(jid m1vox-asr16k): Aloe asr-w15973 FT demo (front :8865, URL_asr16k.txt; weights node-local /scratch)"; done
  for g in 0 1; do echo "slinky-18 $g aloe-demo-11k   # m1vox-11k $(jid m1vox-11k): Aloe w-11488-init FT demo (11k endpoint), instance 1 (+ front :8825 / LB / tunnel, URL_11k.txt)"; done
  for g in 2 3; do echo "slinky-18 $g aloe-demo-11k   # m1vox-11k2 $(jid m1vox-11k2): instance 2"; done
  for g in 4 5; do echo "slinky-18 $g aloe-demo-11k   # m1vox-11k3 $(jid m1vox-11k3): instance 3"; done
} >> $RG
echo "[$(pt)] RESERVED_GPUS: slinky-7 / slinky-18 demo lines + chat-cli re-added"
# ---- verify: new URL in each URL file (!= the pre-restore one), HTTP 200; Aloe demos: "MODEL ready" in the job log
echo "[$(pt)] waiting for new tunnel URLs + readiness (up to 30 min)"
OUT=$N/RESTORED.json; echo "[" > $OUT.tmp; first=1
for r in "${RES[@]}"; do
  IFS='|' read -r lab j node got uf olog <<< "$r"
  u=""; code=""; ready="n/a"
  NW=360; [ "$lab" = m1vox-asr16k ] && NW=2520
  for i in $(seq $NW); do
    u=$(cat $uf 2>/dev/null | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
    if [ -n "$u" ] && [ "$u" != "${OLD[$uf]}" ]; then code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u"); [ "$code" = 200 ] && break; fi
    sleep 5
  done
  got=$(scontrol show job -d $j 2>/dev/null | grep -oE "gpu[^ ]*IDX:[0-9,-]+" | grep -oE "IDX:[0-9,-]+" | head -1)
  if [ "$olog" != "-" ] && [ -n "$olog" ]; then
    ready=no; for i in $(seq 240); do grep -q "MODEL ready" "$olog" 2>/dev/null && { ready=yes; break; }; sleep 5; done
  fi
  echo "[$(pt)] $lab job $j $node $got URL $u HTTP $code (old ${OLD[$uf]}) model_ready $ready"
  [ $first = 1 ] || echo "," >> $OUT.tmp; first=0
  printf '{"name": "%s", "job": "%s", "node": "%s", "gpus": "%s", "url_file": "%s", "url": "%s", "old_url": "%s", "http": "%s", "model_ready": "%s"}' \
    "$lab" "$j" "$node" "$got" "$uf" "$u" "${OLD[$uf]}" "$code" "$ready" >> $OUT.tmp
done
echo "]" >> $OUT.tmp; mv $OUT.tmp $OUT
echo "[$(pt)] restore done -> $OUT"

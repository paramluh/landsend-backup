#!/bin/bash
#SBATCH -J m1vox-live
#SBATCH -N1 -n1 -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=240G
#SBATCH -t 3-00:00:00
#SBATCH -o demo-%j.out
# ALOE VOXTRAL LIVE DEMO (aloe-m1vox-ft, job 22335): model server (2 GPUs, backend :8786) + always-up no-auth front proxy (:8785, injects
# ~/.s4live_token upstream) + cloudflared quick tunnel -> :8785, all in restart loops inside this job. Separate from s4live (22232) and
# t2v-server: own dir, ports, tunnel, URL.txt. The quick-tunnel URL is NEW on every start: aloe-m1vox-live/URL.txt (plain URL, no token).
# CK = the checkpoint served at start; the server hot-swaps in-process (between sessions) to the newest COMPLETE dir in PPMLIVE_SWAP_TO.
#   sbatch demo.sbatch                       (default: start on export-366 if COMPLETE, else the test path = PPM init 11488 + ASR adapter)
#   DRY=1 bash demo.sbatch                   (checks only)
set -u
# RELEASE: source the repo's aloe.env BEFORE sbatch (--export=ALL carries it in; sbatch runs a spooled copy of this script, so it
# cannot find aloe.env by its own path), or pass ALOE_ENV=<repo>/aloe.env. CK defaults to aloe.env ALOE_WEIGHTS.
if [ -z "${ALOE_ROOT:-}" ]; then source "${ALOE_ENV:?source aloe.env first or set ALOE_ENV}"; fi
H=${ALOE_ROOT}/ppm/aloe-m1vox-live
FINAL=${ALOE_ROOT}/ppm/aloe-m1vox-ft/export-366
INIT=${ALOE_ROOT}/ppm/ckpt-infer/ppm-v1/local-11488
ADAPTER=${ALOE_ROOT}/ppm/asr-adapter/ckpt-final/asrppm-nonorm/final-s1972/adapter-bf16-w1972
CK=${CK:-${ALOE_WEIGHTS:-$INIT}}
export PPMLIVE_SWAP_TO=${PPMLIVE_SWAP_TO:-$CK}
export PPMLIVE_DEFAULTS=${PPMLIVE_DEFAULTS:-nopersona}
TOK=${ALOE_TOKEN_FILE}
CF=${CLOUDFLARED:-cloudflared}
FRONT=${FRONT:-8785}; BACK=${BACK:-8786}
# 10-17 INSTANCE 2 (independent copy, same code + checkpoint): per-instance ports / state dir (LOADED.json, SWAP_TO.txt, sessions/) / URL file,
#   all defaulting to instance 1's values. See RESTART_CMD.txt for the instance-2 sbatch line.
export PPMLIVE_STATE_DIR=${PPMLIVE_STATE_DIR:-$H}
URLF=${URLF:-$H/URL.txt}
ID=${SLURM_JOB_ID:-dry}
source $H/env_live.sh

checks() {
  local ok=1
  [ -f "$CK/COMPLETE" ] && echo "ok  checkpoint $CK (COMPLETE)" || { echo "BAD checkpoint $CK has no COMPLETE"; ok=0; }
  [ -s "$TOK" ] && [ "$(stat -c %a "$TOK")" = 600 ] && echo "ok  token file (mode 600)" || { echo "BAD token file $TOK"; ok=0; }
  [ -x "$CF" ] && echo "ok  cloudflared" || { echo "BAD cloudflared $CF"; ok=0; }
  for f in server.py live_ppm.py prompts_ppm.py proxy.py a_level.py index.html levelfix/prior.json; do [ -f "$H/$f" ] || { echo "BAD missing $H/$f"; ok=0; }; done
  $PY -m py_compile $H/server.py $H/live_ppm.py $H/prompts_ppm.py $H/proxy.py $H/a_level.py && echo "ok  py_compile" || ok=0
  $PY -m pyflakes $H/server.py $H/live_ppm.py $H/prompts_ppm.py $H/proxy.py >/dev/null 2>&1 && echo "ok  pyflakes" || echo "warn pyflakes notes (see: $PY -m pyflakes ...)"
  [ $ok = 1 ]
}
if [ "${DRY:-0}" = 1 ]; then checks && echo "== DRY: checks passed" || echo "== DRY: checks FAILED"; exit 0; fi
checks || { echo "ABORT: checks failed"; exit 1; }
mkdir -p $PPMLIVE_STATE_DIR/sessions $H/logs
nvidia-smi -L; echo "CVD=${CUDA_VISIBLE_DEVICES:-} swap targets: $PPMLIVE_SWAP_TO ports front $FRONT back $BACK state $PPMLIVE_STATE_DIR url $URLF"
( while true; do $PY -u $H/proxy.py --port $FRONT --backend $BACK --token-file $TOK --sessions $PPMLIVE_STATE_DIR/sessions >> $H/logs/proxy-$ID.log 2>&1; sleep 0.5; done ) &
( while true; do
    L=$H/logs/cloudflared-$ID-$(date +%s).log
    $CF tunnel --no-autoupdate --url http://127.0.0.1:$FRONT > $L 2>&1 &
    CP=$!
    for i in $(seq 90); do u=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L 2>/dev/null | head -1); [ -n "$u" ] && { echo "$u" > $URLF; echo "tunnel URL $u $(date)"; break; }; sleep 2; done
    wait $CP; sleep 5
  done ) &
while true; do
  AD=""; [ "$CK" = "$INIT" ] && AD="--adapter $ADAPTER"
  $PY -u $H/server.py --ck "$CK" $AD --port $BACK --token-file $TOK
  echo "server exited rc=$? $(date)"; sleep 10
done

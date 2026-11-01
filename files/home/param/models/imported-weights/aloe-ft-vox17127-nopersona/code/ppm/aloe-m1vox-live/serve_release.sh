#!/bin/bash
# RELEASE (HF InternalCan/aloe-ft-vox17127-nopersona): serve "Aloe FT (no persona)" on THIS machine (2 x 80 GB GPUs) without Slurm or a tunnel:
# model server (backend, token-checked) + the always-up front proxy (injects the token upstream, so the page URL needs none), both in
# restart loops, exactly as the live demo's demo.sbatch runs them (minus the cloudflared quick tunnel). Defaults = profile 'nopersona'
# (= inference_defaults.json); every knob stays a live control on the page. EVERY path / port comes from the repo's aloe.env:
#   ALOE_WEIGHTS ALOE_TOKEN_FILE ALOE_STATE ALOE_FRONT_PORT ALOE_BACK_PORT ALOE_PRESET_BANK ALOE_QUIET_BANK VOXTRAL_DIR ALOE_MIMI_DIR ALOE_FA3_DIR ALOE_PY
#   bash code/ppm/aloe-m1vox-live/serve_release.sh          then open http://127.0.0.1:$ALOE_FRONT_PORT/
# The token file is created (random, mode 600) if missing.
set -u
H=$(dirname "$(readlink -f "$0")")
source "$H/env_live.sh"                                      # sources aloe.env (repo root) unless ALOE_ROOT is already set
for v in ALOE_WEIGHTS ALOE_TOKEN_FILE ALOE_STATE ALOE_FRONT_PORT ALOE_BACK_PORT ALOE_PRESET_BANK ALOE_QUIET_BANK VOXTRAL_DIR ALOE_MIMI_DIR ALOE_FA3_DIR; do
  [ -n "${!v:-}" ] || { echo "ABORT: $v is not set (aloe.env)"; exit 1; }; done
[ -f "$ALOE_WEIGHTS/COMPLETE" ] && [ -f "$ALOE_WEIGHTS/.metadata" ] || { echo "ABORT: $ALOE_WEIGHTS is not a COMPLETE consolidated DCP export"; exit 1; }
[ -f "$ALOE_WEIGHTS/vgain_theta_fp32.json" ] || echo "WARN: no vgain_theta_fp32.json next to the weights: s falls back to exp(bf16 theta) (0.31 % low)"
for f in "$ALOE_PRESET_BANK/presets.json" "$ALOE_QUIET_BANK" "$VOXTRAL_DIR/audio_tower.safetensors" "$ALOE_MIMI_DIR/model.safetensors" \
         "$ALOE_FA3_DIR/build/torch28-cxx11-cu128-x86_64-linux/__init__.py" "$ALOE_PY"; do
  [ -e "$f" ] || { echo "ABORT: missing $f (see QUICKSTART.md)"; exit 1; }; done
if [ ! -s "$ALOE_TOKEN_FILE" ]; then ( umask 077; $ALOE_PY -c 'import secrets; print(secrets.token_hex(16))' > "$ALOE_TOKEN_FILE" ); fi
export PPMLIVE_STATE_DIR=$ALOE_STATE PPMLIVE_SWAP_TO=$ALOE_WEIGHTS PPMLIVE_DEFAULTS=${PPMLIVE_DEFAULTS:-nopersona} PPMLIVE_BIND=${PPMLIVE_BIND:-127.0.0.1}
mkdir -p "$ALOE_STATE/sessions" "$ALOE_STATE/logs"
[ -e "$ALOE_STATE/clips" ] || ln -s "$ALOE_PRESET_BANK/clips" "$ALOE_STATE/clips"      # preset voice-clip previews served by the proxy
( while true; do $PY -u "$H/proxy.py" --port "$ALOE_FRONT_PORT" --backend "$ALOE_BACK_PORT" --token-file "$ALOE_TOKEN_FILE" --sessions "$ALOE_STATE/sessions" \
    >> "$ALOE_STATE/logs/proxy.log" 2>&1; sleep 0.5; done ) &
echo "front proxy http://127.0.0.1:$ALOE_FRONT_PORT/  (backend :$ALOE_BACK_PORT, state $ALOE_STATE, defaults $PPMLIVE_DEFAULTS); load + graph capture ~1.5-2.5 min"
while true; do
  $PY -u "$H/server.py" --ck "$ALOE_WEIGHTS" --port "$ALOE_BACK_PORT" --token-file "$ALOE_TOKEN_FILE"
  echo "server exited rc=$? $(date)"; [ "${ALOE_SERVE_ONCE:-0}" = 1 ] && break; sleep 10
done

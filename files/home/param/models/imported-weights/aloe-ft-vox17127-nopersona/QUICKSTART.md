# Quickstart: serve "Aloe FT (no persona)" on a fresh 2-GPU box

This takes you from an empty machine to a live talking page and a scripted test session. Every path and port comes from one file,
`aloe.env`.

**Requirements**
- Linux x86_64 with **2 x 80 GB Hopper GPUs** (H100). The FA3 kernels are Hopper-only.
- An NVIDIA driver for CUDA 12.8.
- About 90 GB of disk.
- `uv`, and the Python 3.10 headers (`python3.10-dev`), which Triton needs to build its kernel launchers.
- An HF login that can read the private `InternalCan` org.

## 1. Clone

```bash
hf download InternalCan/aloe-ft-vox17127-nopersona --local-dir aloe-ft-vox17127-nopersona    # ~80.5 GB (78.4 GB weights)
cd aloe-ft-vox17127-nopersona
```

## 2. Python environment

```bash
source aloe.env                                    # every path below comes from here (ALOE_PY = $ALOE_HOME/venv/bin/python)
uv venv "$ALOE_HOME/venv" --python 3.10
uv pip install --python "$ALOE_PY" -r env/pip_freeze.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
sudo apt-get install -y python3.10-dev             # or set ALOE_PY_INCLUDE in aloe.env to unpacked libpython3.10-dev headers
```

`env/pip_freeze.txt` lists the exact versions the live demo runs: Python 3.10.12, torch 2.8.0+cu128, triton 3.4.0 and transformers
5.13.0. `env/torch_cuda.txt` has the CUDA and cuDNN versions.

## 3. Download the two public assets, pinned

The repo already ships the Voxtral audio tower and configs, the quiet-frames bank, the 12 voice-clip presets and the Qwen3-32B
tokenizer. Only two assets come from public HF:

```bash
source aloe.env
# Mimi codec (kyutai/mimi), exact revision
hf download kyutai/mimi --revision 89091b3e466eb6a9d11e537bf26b144f194978f7 \
    config.json model.safetensors preprocessor_config.json --local-dir "$ALOE_MIMI_DIR"
# FlashAttention-3 kernels (kernels-community/vllm-flash-attn3, branch v1), exact revision, torch 2.8 / CUDA 12.8 build only
hf download kernels-community/vllm-flash-attn3 --revision 94e47890779e2a04d4994695f54a48057f0e8177 \
    --include "build/torch28-cxx11-cu128-x86_64-linux/*" --local-dir "$ALOE_FA3_DIR"
```

Check every file the server will load against the pins: the weights' `SHA256SUMS` and `env/ASSETS_PINNED.json`.

```bash
"$ALOE_PY" -I code/release_tools/verify_assets.py          # one line per group, then PASS (add --skip-weights to skip the 78 GB)
```

## 4. Launch

```bash
bash code/ppm/aloe-m1vox-live/serve_release.sh
```

This starts the model server on `$ALOE_BACK_PORT` and the front proxy on `$ALOE_FRONT_PORT`, both in restart loops. It also creates
a random backend token in `$ALOE_TOKEN_FILE` if there isn't one; the proxy injects it, so the page needs no token.

The first start takes about 1.5-2.5 min: loading the weights (~75 s), capturing the CUDA graphs and a smoke test. It is ready when
`$ALOE_STATE/LOADED.json` appears and `curl -s http://127.0.0.1:$ALOE_FRONT_PORT/api/presets` reports `"state": "ready"`. The
defaults are the release config (`inference_defaults.json`): profile `nopersona`, speaker preset "Luke", voice clip only, no persona.

## 5. Talk to it

- **In a browser:** open `http://127.0.0.1:8985/` (from a laptop: `ssh -L 8985:127.0.0.1:8985 <gpu host>`). Use headphones. Every
  setting is a live control and is kept in the page URL.
- **Scripted test client**, with any 24 kHz mono wav as the caller (`ffmpeg -i in.any -ac 1 -ar 24000 caller.wav`):

  ```bash
  source aloe.env && source code/ppm/aloe-m1vox-live/env_live.sh
  $PY -I code/ppm/aloe-m1vox-live/test_client.py http://127.0.0.1:$ALOE_FRONT_PORT caller.wav out 30
  # out/result.json: frames, B audio seconds, B's own word lane ('words'), per-frame ms p50 / p95; out/b.wav: B's audio
  ```

  The test client never takes over a session that is in use.
- **Stop:** Ctrl-C the launcher, or kill its process group.

## Notes

- Expect about 62 ms per 80 ms frame on unthrottled H100s.
- Sessions are capped at 600 s.
- `$ALOE_STATE/sessions/<id>/` keeps each session's audio, tokens and timing.
- To serve under Slurm with a public Cloudflare quick tunnel instead, as the live demo does: `source aloe.env`, then
  `sbatch --gres=gpu:2 -c 14 --mem=100G --export=ALL,FRONT=$ALOE_FRONT_PORT,BACK=$ALOE_BACK_PORT,PPMLIVE_STATE_DIR=$ALOE_STATE,URLF=$ALOE_STATE/URL.txt code/ppm/aloe-m1vox-live/demo.sbatch`.
  This needs `cloudflared` (`CLOUDFLARED` in `aloe.env`) and a 600-mode `$ALOE_TOKEN_FILE`. The public URL is written to
  `$ALOE_STATE/URL.txt`.
- For the architecture, prompt format, sampling table, training recipe and eval, see `README.md`.

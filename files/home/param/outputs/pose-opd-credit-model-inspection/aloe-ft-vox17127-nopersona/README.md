---
library_name: pytorch
tags:
- speech
- full-duplex
- spoken-dialogue
- voxtral
- mimi
---

# Aloe FT (no persona): vox17127, export-249

**PRIVATE, internal (InternalCan).** Full-duplex speech-to-speech model. It is the Aloe fine-tune that the live demo shows as
**"Aloe FT (no persona)"**: Slurm job 29345, run `ppm-aloe-samsep-vox17127-j29345`, final export step 249. Its final eval normalised
word CE is **1.8023**.

The model is a PPM 32B stack: a Qwen3-32B backbone (64 layers), a 4B speech tower (36 layers) and a 0.6B depth decoder (28 layers).
Channel A is the caller, heard through the frozen Voxtral Realtime audio encoder plus a trained adapter scaled by a learned gain `s`.
Channel B is the assistant. Every 80 ms frame (12.5 Hz), B generates its text lane (Qwen3 word pieces plus PAD / NEW_WORD timing
tokens) and 32 Mimi codebooks: codebook 0 comes from the speech tower, codebooks 1-31 from the 0.6B decoder, at a code delay of 2.
The B audio is decoded with streaming Mimi. B is prompted with a voice clip only: there is never any persona text.

| | |
|---|---|
| `weights/` | The bf16 weights in the exact layout the live demo loads. See [Weights](#weights). |
| `weights/vgain_theta_fp32.json` | The exact fp32 Voxtral gain. See [Voxtral gain](#voxtral-gain-vgain-use-the-fp32-value). |
| `weights/SHA256SUMS` | sha256 and size of every file in `weights/`. |
| `inference_defaults.json` | The inference config the user set for this model (52 keys). It is the default of the code in this repo. |
| `code/` | A snapshot of the demo server and every local module it imports, plus the trainer code for reference. See [Code](#code). |
| `aloe.env` | **The one config file.** Every path and port the code uses comes from here. |
| `QUICKSTART.md` | Fresh 2-GPU box, end to end: clone, env, downloads, verify, launch, page / test client. |
| `assets/` | Shipped runtime assets: the 12 speaker presets (SAM-isolated voice clips + previews), the quiet-frames bank, and the Voxtral audio tower + configs. |
| `env/` | Python, torch and CUDA versions; `pip_freeze.txt` of the venv the demo runs with; `ASSETS_PINNED.json` (repo ids, revisions, per-file sha256 of every external asset); `FA3_INSTALL.txt`. |
| `SECRET_SCAN.md` | The pre-upload secret-scan summary (no values). |

## Weights

`weights/` holds the **consolidated bf16 torch DCP export** `export-vox17127-249`. It contains `.metadata`, 64 `__<k>_0.distcp`
shards, `COMPLETE` and `train_cfg.json`: 67 files and 78.36 GB in total. Every model tensor is in it, including the trained Voxtral
adapter (`vadapter.*`) and the gain (`vgain.theta`), at step 249.

**Why this layout.** The live "Aloe FT (no persona)" backends serve this exact directory. Their launcher passes
`CK=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox/export-vox17127-249` (the live demo's `a16861/launch_aloe.sh`; in this snapshot it reads `ALOE_WEIGHTS` from `aloe.env`), and
the demo's `LOADED.json` reports `"kind": "dcp"`. The loader is `live_ppm.ckpt_info` / `load_weights`, which calls
`torch.distributed.checkpoint.load(..., no_dist=True)` on `.metadata`.

The demo can also load the world-64 `ppm-local-bf16-shards-v1` copy (`rank*.pt` + `rep.pt` + `meta.json`). That copy is backed up
separately at `InternalCan/ppm-aloe-samsep-vox17127-final/export-249`, so only the consolidated DCP is published here.

The same DCP is also backed up and verified at `InternalCan/ppm-aloe-samsep-vox17127-final/export-249-dcp`.

To check a download:

```bash
cd weights && grep -v '^#' SHA256SUMS | awk '{print $1"  "$3}' | sha256sum -c -
```

### Voxtral gain (vgain): use the fp32 value

The channel-A input is `s x adapter(Voxtral(audio))`, with `s = exp(theta)`.

- **Use theta = -5.621907711029053, so s = 0.0036177329326430177.** This is the trainer's fp32 master value after optimizer step
  249. It comes from the run's end full save, and the run also logged `s` as `train/vox_a_scale` at step 249.
- **Do not use the DCP's stored value.** The DCP stores `vgain.theta` in bf16, and `exp(bf16 theta)` = 0.0036065631, which is
  0.31 % too low.
- **How the live demo gets `s`.** For a DCP export, `live_ppm.build_model` reads `train_cfg.json` to find the run's
  `log_root/run_tag/metrics.jsonl`. It then takes `train/vox_a_scale` at the checkpoint step (`s_from_metrics`) and overwrites
  `vgain.theta` with `log(s)`. The demo's `LOADED.json` shows `"s": 0.0036177329326430177` with `s_src` "trainer fp32 s at step
  249 (.../metrics.jsonl)".
- **How the release code gets `s`.** The metrics file is not part of this release. Instead, `code/ppm/aloe-m1vox-live/live_ppm.py`
  reads `weights/vgain_theta_fp32.json` next to the DCP. It applies the same 0.04 sanity check against the bf16 theta, and the log
  line says `exp(vgain_theta_fp32.json vgain_theta_fp32 -5.621907711)`. `build_model(..., s_value=...)` overrides this, if needed.

## Prompt format

The prompt follows the training layout (`data_aloe_vox.speech_micro_vox`; the trainer's `sysprompt v2_voiceB_only`,
`persona_sides none`).

- **System prompt, positions `[0, P)`, on channel B only.**
  - B gets a **voice clip**: Mimi codes only, with a blank text lane, 3-10 s long. The clip is either a held-out preset speaker's
    own clip, or an upload that is resampled to mono 24 kHz, normalised to -23 LUFS and Mimi-encoded.
  - There is **no persona text block and no context text**. This is the `voice_clip_only` prompt mode, and the model was trained
    without persona text.
  - Channel A never has a voice clip.
  - The prompt is capped at 2,000 positions.
  - With the default preset, `aloe-1-1648-0` (speaker "Luke"), P = 94.
- **Conversation frame f is position P + f.**
  - **Channel A** is the user's live speech. Its lane is PAD text with no Mimi codes and `present_A = 0`. To that, the engine adds
    `s x adapter(Voxtral(A audio of frame f))` on the same frame, with no delay. The **Voxtral latents are computed in the loop**
    from the 24 kHz waveform: a streaming Voxtral Realtime encoder with 3 ms lookahead, matching training's in-loop front up to
    bf16 round-off.
  - **Channel B** is the assistant. It carries the lane token of frame f (word pieces plus PAD / NEW_WORD; EOB is disabled) and
    B's Mimi codes at a code delay of 2, so positions P and P + 1 carry no B codes.
  - The lane grammar applies: after NEW_WORD comes a word piece, and after PAD no word piece follows.
- **In training**, 30 % of rows had a B voice clip (`voice_frac 0.3`) and the rest had none. The demo always gives B a clip.

## Inference defaults (`inference_defaults.json`)

These are the exact settings the user saved for this model on the live demo page (10-25). They are the **defaults of the code in
this repo**. In the code snapshot, the server's `live_ppm.DEFAULTS` is profile `nopersona`, selected when `PPMLIVE_DEFAULTS` is
unset. The page's `RULES` (`NOPERSONA` in `index.html`) and the `CK_LABELS.json` default preset are set to match. Every knob stays a
live control on the page.

Notes on the table:
- `minp` is absolute.
- `cfg_s`, `cfg_cb0` and `cfg_uncond_src` apply only when `cfg_on` = 1.
- `lim_*` apply only when `lim_on` = 1.
- The live demo itself was started with `PPMLIVE_DEFAULTS=unbiased`. There, the user's settings come from the page and its URL.

| key (page / API name) | value | server parameter (internal name) | meaning |
|---|---|---|---|
| factorised_text | 1 | factorised_text | B's text draw in 2 steps: PAD / NEW_WORD / WORD category first, then the word piece |
| cat_temp | 1 | cat_temp | temperature of the category step |
| pad_bias | 0 | pad_bias | static PAD logit bias (ignored while adaptive PAD bias is on) |
| adaptive_pad_on | 1 | adaptive_pad_on | adaptive PAD bias **on** |
| pad_silent_s | 3 | pad_silent_s | B counts as silent after 3 s with no word / NEW_WORD |
| pad_bias_silent | -0.1 | pad_bias_silent | PAD bias while silent (slightly more likely to speak) |
| pad_bias_other | 0 | pad_bias_other | PAD bias otherwise |
| text_temp | 1 | text_temp | word temperature |
| text_topk | 0 | text_topk | word top-k (off) |
| text_topp | 1 | text_topp | word top-p (off) |
| minp | 0.05 | min_p | word min-p |
| minp_mode | absolute | min_p_rel = 0 | min-p mode |
| freq_pen | 0.3 | freq_pen | frequency penalty on word pieces |
| freq_win | 50 | freq_win | frequency-penalty window: B's last 50 word pieces |
| text_rep | 1 | text_rep | text repetition penalty (off) |
| text_rep_win | 64 | text_rep_win | its window |
| cb0_temp | 1 | cb0_temp | audio codebook 0 temperature |
| cb0_topk | 99 | cb0_topk | audio codebook 0 top-k |
| light_temp | 1 | light_temp | codebooks 1-31 temperature |
| light_topk | 79 | light_topk | codebooks 1-31 top-k |
| audio_rep | 1 | audio_rep_pen | audio repetition penalty (off) |
| audio_rep_win | 64 | audio_rep_win | its window (frames) |
| adaptive_rep_on | 0 | adaptive_rep_on | adaptive repetition penalty (off) |
| speak_window_frames | 25 | speak_window_frames | "speaking" window for the adaptive repetition penalty |
| audio_rep_speaking | 1 | audio_rep_speaking | (adaptive rep only) |
| audio_rep_silent | 1 | audio_rep_silent | (adaptive rep only) |
| text_rep_speaking | 1 | text_rep_speaking | (adaptive rep only) |
| text_rep_silent | 1 | text_rep_silent | (adaptive rep only) |
| cfg_on | 0 | cfg_on | CFG off |
| cfg_s | 1.5 | cfg_s | CFG strength (only with cfg_on) |
| cfg_cb0 | 0 | cfg_cb0 | CFG on cb0 too (only with cfg_on) |
| cfg_uncond_src | quiet_frames | cfg_uncond_src (start param) | what the CFG uncond stream hears |
| seed | random | seed = 0 (start param) | 0 means a random seed at every Start |
| max_len_s | 600 | max_s (start param) | max session length (s) |
| a_gain | 0 | a_gain_db (host param) | manual mic gain (dB) |
| a_lufs_on | 0 | a_lufs_on (host param) | -23 LUFS mic loudness match (off: mic as captured) |
| lim_on | 0 | limit_on (host param) | B output limiter (off) |
| lim_ceiling | 0.98 | limit_ceil | limiter ceiling (only with lim_on) |
| lim_attack_ms | 2 | limit_attack_ms | (only with lim_on) |
| lim_release_ms | 50 | limit_release_ms | (only with lim_on) |
| allow_eob | 0 | allow_eob | EOB in B's lane disabled, as in training |
| free_text | 0 | free_text | lane grammar on |
| word_greedy | 0 | word_greedy | off |
| a_target_lufs | -23 | a_target_lufs (host param) | (only with a_lufs_on) |
| a_slew_db_s | 3 | a_slew_db_s (host param) | (only with a_lufs_on) |
| a_g0_db | 0 | a_g0_db (start param) | A gain before the first voiced frame (0 = automatic) |
| model | aloe | none: front proxy | picks the "Aloe FT (no persona)" backends on the multi-model demo proxy (`lb.json` model id) |
| preset | aloe-1-1648-0 | none: `CK_LABELS.json` `default_preset` | B speaker preset "Luke" (held-out call), first in the list and the server default |
| clip | preset:aloe-1-1648-0 | none: the default preset's own clip | B voice clip |
| persona | none | none: prompt mode `voice_clip_only` | no persona text |
| aec | 1 | none: page-side | browser echo cancellation (getUserMedia constraint) |
| ns | 1 | none: page-side | browser noise suppression (getUserMedia constraint) |

46 of the 52 keys are server parameters, and the snapshot server's in-effect defaults equal them, as checked by a script. The 6 keys
without a server parameter are `model`, `preset`, `clip`, `persona`, `aec` and `ns`. `preset`, `clip` and `persona` are covered by
`CK_LABELS.json`: the `default_preset` and `voice_clip_only` prompt. `model` is the demo front proxy's routing. `aec` and `ns` are the
browser's mic constraints, and the page's checkboxes default to on.

**The live demo page, preloaded with these parameters** (no token needed, because the front proxy injects it):

https://webster-twist-own-growing.trycloudflare.com/?factorised_text=1&cat_temp=1&pad_bias=0&adaptive_pad_on=1&pad_silent_s=3&pad_bias_silent=-0.1&pad_bias_other=0&text_temp=1&text_topk=0&text_topp=1&minp=0.05&minp_mode=absolute&freq_pen=0.3&freq_win=50&text_rep=1&text_rep_win=64&cb0_temp=1&cb0_topk=99&light_temp=1&light_topk=79&audio_rep=1&audio_rep_win=64&adaptive_rep_on=0&speak_window_frames=25&audio_rep_speaking=1&audio_rep_silent=1&text_rep_speaking=1&text_rep_silent=1&cfg_on=0&cfg_s=1.5&cfg_cb0=0&cfg_uncond_src=quiet_frames&seed=random&max_len_s=600&a_gain=0&a_lufs_on=0&lim_on=0&lim_ceiling=0.98&lim_attack_ms=2&lim_release_ms=50&allow_eob=0&free_text=0&word_greedy=0&a_target_lufs=-23&a_slew_db_s=3&a_g0_db=0&model=aloe&preset=aloe-1-1648-0&clip=preset:aloe-1-1648-0&persona=none&aec=1&ns=1

The page reads every setting from its query string (`index.html` `urlRead`: URL > saved settings > defaults). It is a Cloudflare
quick tunnel, so the host changes whenever the demo's front job restarts. The current host is in
`/data/home/amartyavarma/ppm/aloe-m1vox-live/URL.txt` on the cluster; keep the query string.

## How to load and serve

**Hardware.** Serving needs 2 x 80 GB GPUs (H100). Loading the weights and capturing the CUDA graphs (max session 600 s) takes
about 75 s, plus a start-up smoke test. GPU memory after capture is about 58.7 / 33.4 GiB. The backbone is split 44 / 20 layers. The last GPU also holds the head, speech
tower and 0.6B decoder; the first GPU holds the Voxtral front, the adapter and the streaming Mimi decoder. Frame latency is about
62 ms p50 against the 80 ms real-time budget.

**Software.** See `env/`: Python 3.10.12, torch 2.8.0+cu128 (CUDA 12.8), triton 3.4.0 and transformers 5.13.0. bf16 only.

**Step by step: [QUICKSTART.md](QUICKSTART.md).** It covers clone, venv, the two pinned downloads, verify, launch, and opening the
page or running the test client on a fresh 2-GPU box.

### One config file: `aloe.env`

Every path and port the code uses comes from `aloe.env` at the repo root. `code/` has no hard-coded machine paths:
`grep -rnE "/data/home|/scratch/|/home/" code/` is empty.

`code/` mirrors the original source tree (`code/ppm/...`, `code/qwen-1ch/...`) rooted at `$ALOE_ROOT`. Python modules read
`ALOE_ROOT` / `ALOE_SCRATCH` and the asset variables below; shell scripts use `${ALOE_ROOT}`. Source it first: `source aloe.env`.

| variable | default | what |
|---|---|---|
| `ALOE_WEIGHTS` | `weights/` | the DCP + `vgain_theta_fp32.json` |
| `ALOE_PRESET_BANK` | `assets/presets/` | **shipped**: the 12 B speaker presets the live demo serves (offline bank, below) |
| `ALOE_QUIET_BANK` | `assets/quiet_bank_m2.npz` | **shipped** (48 MB, the exact file the demo loads): quiet A frames for the CFG uncond stream, loaded at engine init |
| `VOXTRAL_DIR` | `assets/voxtral-realtime/` | **shipped**: the frozen Voxtral Realtime audio tower (`audio_tower.safetensors`, not a public file) + its 4 config files |
| `ALOE_MIMI_DIR` | `assets/mimi/` | **download**: `kyutai/mimi` @ `89091b3e466eb6a9d11e537bf26b144f194978f7` |
| `ALOE_FA3_DIR` | `assets/vllm-flash-attn3/` | **download**: `kernels-community/vllm-flash-attn3` v1 @ `94e47890779e2a04d4994695f54a48057f0e8177`, build `torch28-cxx11-cu128-x86_64-linux` (`env/FA3_INSTALL.txt`) |
| `ALOE_QWEN_DIR` | `code/models/Qwen3-32B` | **shipped**: Qwen3-32B tokenizer + config only (Qwen/Qwen3-32B @ 9216db57) |
| `ALOE_PY`, `ALOE_PY_INCLUDE` | `venv/bin/python`, empty | the venv built from `env/pip_freeze.txt`; Python 3.10 headers for Triton (empty = system `python3.10-dev`) |
| `ALOE_TOKEN_FILE`, `ALOE_STATE`, `ALOE_FRONT_PORT`, `ALOE_BACK_PORT` | `token`, `state/`, 8985, 8986 | serving |

**Pins (`env/ASSETS_PINNED.json`).** For every public file the server loads, this records the repo id, the exact revision, and the
per-file size and sha256. Each was checked against the HF API metadata of that revision: the LFS sha256 for large files, the git blob
sha1 for small ones. All match.
- **Voxtral configs:** `mistralai/Voxtral-Mini-4B-Realtime-2602` @ `2769294da9567371363522aac9bbcfdd19447add`.
- **Voxtral audio tower:** shipped. It is cut from that revision's `model.safetensors` (LFS sha256 `e745e490...`) by
  `code/qwen-1ch/stage4/voxtral_tools/fetch_voxtral_encoder.py`. All 421 `audio_tower.*` tensors match the upstream header in name,
  dtype, shape and byte length.
- **Mimi:** `kyutai/mimi` @ `89091b3e`.
- **FA3:** `kernels-community/vllm-flash-attn3` v1 @ `94e47890`. The cluster copy was fetched at `905da400`, which is no longer on the
  Hub; v1 holds identical files.
- **Qwen3-32B tokenizer:** @ `9216db57`.

The shipped assets' sha256 are in its `shipped` section. `code/release_tools/verify_assets.py` checks all of it, weights included.

**Offline preset bank (`assets/presets/`).** The live demo builds its 12 speaker presets from internal data stores. The release ships
exactly what those presets use: for each preset, its id, the speaker's name, the clip window and the **SAM-isolated Mimi codes of
the B voice clip** (audio only, as served), plus the 24 kHz preview wavs the page plays. It also stores the system-prompt arrays the
cluster built for each preset. No persona text or other training data is included.
- `prompts_ppm.BankPresets` is selected by `ALOE_PRESET_BANK`. Its start-up selftest rebuilds every preset's prompt and requires it
  to equal the cluster's, ids / depth / present / P.
- When the bank was built, a prompt from bank codes was checked equal to one from the data store for all 12 presets.
- The page labels a preset "Luke · held-out Aloe speaker", where the live demo shows a blurb from the persona text.

### Serve

```bash
source aloe.env
bash code/ppm/aloe-m1vox-live/serve_release.sh        # backend :$ALOE_BACK_PORT + front proxy :$ALOE_FRONT_PORT; page http://127.0.0.1:8985/
```

The backend is `server.py --ck $ALOE_WEIGHTS` with `PPMLIVE_DEFAULTS=nopersona`; the front proxy is `proxy.py`. Both run in
restart loops, as in the demo's `demo.sbatch`. The proxy serves the preset previews from `$ALOE_PRESET_BANK/clips`.

To serve the live-demo way instead (a Slurm job plus a public Cloudflare quick tunnel; adapted from `a16861/launch_aloe.sh`, which ran
3 such backends):

```bash
source aloe.env
sbatch --gres=gpu:2 -c 14 --mem=100G -o $ALOE_STATE/demo-%j.out \
  --export=ALL,FRONT=$ALOE_FRONT_PORT,BACK=$ALOE_BACK_PORT,PPMLIVE_STATE_DIR=$ALOE_STATE,URLF=$ALOE_STATE/URL.txt code/ppm/aloe-m1vox-live/demo.sbatch
```

`CK` defaults to `$ALOE_WEIGHTS` and `PPMLIVE_DEFAULTS` to `nopersona`. This needs `cloudflared` and a 600-mode `$ALOE_TOKEN_FILE`.
The live run also set `PPMLIVE_DEFAULTS=unbiased` (the page then carries the user's settings) and `PPMLIVE_BIND=0.0.0.0` (a front
proxy on another node).

### Test client (non-preempting) and the Python API

```bash
source aloe.env && source code/ppm/aloe-m1vox-live/env_live.sh     # sets PY
$PY -I code/ppm/aloe-m1vox-live/test_client.py http://127.0.0.1:$ALOE_FRONT_PORT <24 kHz mono wav> <out dir> [secs]   # server defaults = this config
```

To use the engine directly, follow the pattern in `server.py` (`Model`, `Session._run`, `Server.smoke`):

```python
import live_ppm as LP, prompts_ppm as PR, torch                 # after: source aloe.env; sys.path += [code/ppm/aloe-m1vox-live]
devs = [torch.device("cuda:0"), torch.device("cuda:1")]
m, ldev, cfgs, info = LP.build_model(os.environ["ALOE_WEIGHTS"], devs)   # s from weights/vgain_theta_fp32.json
eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=600.0); eng.capture()
pres = PR.make_presets(codes="samsep", first="aloe-1-1648-0")  # the offline bank (ALOE_PRESET_BANK)
pre, _ = PR.build_prefix("", pres.items[0]["clip"])           # voice clip only, no persona
eng.start(pre, seed, {})                                      # {} -> LP.DEFAULTS (= inference_defaults.json)
# per 80 ms frame f: eng.write_a(f * 1920, mic[f*1920 : (f+1)*1920 + LP.VS.LOOK24]); dec = eng.step(eng.P + f)
#                    tok, p_pad, p_nw, b_audio = eng.result(eng.P + f, dec)
```

## Training recipe (job 29345)

- **Init.** The PPMVOX v3 stop export `w-17127`, from private HF `InternalCan/ppmvox-v3-16861`. That is the PPM 32B with its own
  jointly trained Voxtral adapter and fp32 vgain (theta from its `meta.json`; s ≈ 0.003651 at step 0). With `ALOE_INIT_VOX=1`,
  there is no separate ASR-adapter load and no s recalibration.
- **Recipe.** Job **28511**'s recipe (a16861w480) with only the init changed. 28511 is jobs 26232 / 26467 minus the co-train.
  The resolved-config diff against 28511 is run dirs, hosts, init / adapter init and export paths only (`cfg_diff_28511.txt`).
- **Data.** Aloe handover rows (Model-1 handover, none-crossbleed bucket), first chunks only. The Mimi codes on both lanes are
  swapped for the **SAM-separated (isolated)** streams. 122 purged conversations are dropped. That gives 15,898 train micros and
  316 eval micros.
  - Channel A is Voxtral latents of the partner's **raw-gain** FLAC, computed in the training loop (`VOX_A_GAIN=raw`). It has no A
    Mimi codes and no A text.
  - Channel B is the text lane with NEW_WORD timing plus Mimi codes.
  - Both orientations are used.
  - Prompt: a B voice clip on 30 % of rows, no persona.
- **Length.** 249 optimizer steps = 1 epoch.
  - Logical world 64 on 8 nodes x 8 H100 (1 micro per GPU per step, no host offload).
  - Schedule: warmup 8, then linear (WSD) decay to 0 at step 249.
  - AdamW beta2 0.95, bf16.
  - Eval every 25 steps plus the export steps (41, 83, 124, 166, 208).
  - Wall clock: 07:22-07:52 PT, 10-24.
- **LRs** (`code/ppm/aloe-ft-32-samsep/vox/run_vox.env`):

  | group | LR | grad clip |
  |---|---|---|
  | trunk | 3e-6 | 0.5 |
  | spec | 6e-5 | 1.0 |
  | proj | 3e-5 | 1.0 |
  | audio | 6e-5 | 1.0 |
  | tower_s | 1.5e-5 | 1.0 |
  | vadapter | 5e-4 | 1.0 |
  | vgain | 5e-4 | 1.0 |
  | 0.6B decoder | 1.5e-5 | 1.0 |

  The decoder runs K=1 at 300 micro frames.
- **Loss.** B text CE x 5. It is factorised into word / category 1, PAD timing 1, NEW_WORD 1 and EOB 1. To that are added cb0 CE x 1
  and the decoder's cb1-31 CE x 1.
- **Launch.** `QQA_ENV=code/ppm/aloe-ft-32-samsep/vox/run_vox.env sbatch -N8 -w <8 hosts> --ntasks-per-node=8 --cpus-per-task=6
  code/ppm/aloe-ft-32-samsep/vox/run_vox.sbatch`. Per rank, this runs `code/task_samsep.sh`, which calls `train_samsep_vox.py`.
  The init is staged per host by `vox/stage_init.sbatch`.

## Eval (standard Aloe eval, 316 held-out micros)

The headline metric is **normalised word CE** for B (lower is better).

| | normalised word CE |
|---|---|
| step 0 (init) | 1.8484 |
| **final, step 249** | **1.8023** |
| for comparison: job 28511 (a16861w480, same recipe on the pure-PPM base + ASR adapter), final | 1.8074 |

The columns below are:
- **all**: `eval_speech/word_ce_norm`.
- **asstB**: B = the call's assistant side.
- **userB**: B = the call's user side.
- **cb0_ce** and **light_ce**: B's audio codebook 0 CE and the CE of codebooks 1-31.

| step | all | asstB | userB | cb0_ce | light_ce |
|---|---|---|---|---|---|
| 0 | 1.8484 | 1.8656 | 1.8308 | 1.0637 | 2.5037 |
| 25 | 1.8265 | 1.8441 | 1.8083 | 1.0494 | 2.4900 |
| 50 | 1.8172 | 1.8352 | 1.7986 | 1.0452 | 2.4844 |
| 100 | 1.8098 | 1.8278 | 1.7913 | 1.0419 | 2.4798 |
| 150 | 1.8056 | 1.8239 | 1.7869 | 1.0402 | 2.4774 |
| 200 | 1.8036 | 1.8217 | 1.7850 | 1.0393 | 2.4761 |
| **249** | **1.8023** | **1.8205** | **1.7836** | **1.0390** | **2.4756** |

All values are from `runs/ppm-aloe-samsep-vox17127-j29345/metrics.jsonl` on the cluster.

## Code

`code/` mirrors the original source tree (`code/ppm/...`, `code/qwen-1ch/...`), rooted at `$ALOE_ROOT`. Release-wide changes to every
file:
- Each hard-coded cluster path is replaced by the `aloe.env` variables. In Python, string literals become f-strings on
  `_ALOE_ROOT` / `_ALOE_SCRATCH` (read from `ALOE_ROOT` / `ALOE_SCRATCH`), which gives the same values on the cluster. In shell,
  they become `${ALOE_ROOT}` / `${ALOE_SCRATCH}`.
- Asset locations read `ALOE_QWEN_DIR`, `VOXTRAL_DIR`, `ALOE_MIMI_DIR`, `ALOE_FA3_DIR` and `ALOE_QUIET_BANK`.

- **Demo server** (`ppm/aloe-m1vox-live/`). Copies of the live files, with the release changes marked `RELEASE SNAPSHOT`:
  - `server.py`: an unlisted checkpoint takes the `"*"` entry of `CK_LABELS.json`; presets come through `prompts_ppm.make_presets`.
  - `live_ppm.py`: the `nopersona` default profile, and the `vgain_theta_fp32.json` loader.
  - `prompts_ppm.py`: `BankPresets` and the offline selftests.
  - `index.html`: the page's `NOPERSONA` defaults.
  - `CK_LABELS.json`: `"*"`, voice clip only, sam-sep presets, `default_preset` `aloe-1-1648-0`.
  - `env_live.sh` and `demo.sbatch`: driven by `aloe.env`.
  - `serve_release.sh`: new.

  Otherwise unchanged: `a_level.py`, `proxy.py`, `levelfix/prior.json`, `a16861/launch_aloe.sh`, `make_clip_wavs.py` (needs the
  cluster stores) and `test_client.py`.
- **Modules the demo imports** (found by tracing its imports, transitively, then statically checking lazy imports):
  - `ppm/aloe-m1vox-ft/code/` (`model_ppm.py`, `data_*.py`, `rowcache_vox.py`, `lib/`, `vendor/`)
  - `ppm/code/infer_ppm.py`
  - `qwen-1ch/infer/s4fast/vox_stream.py`, `s4self/vox_fast.py`, `s4self/mimi_dec_stream.py`, `s3fast/light_kernels.py`
  - `qwen-1ch/stage4/voxtral_front.py` and `vadapter_s4q.py`
  - `qwen-1ch/common/live_params_schema.json`
  - `ppm/aloe-v2vox-ft/code/data_aloe_v2.py` (cluster prompt selftest)
  - `ppm/configs/Qwen3-4B/config.json`
  - `qwen-s2-imend/hera3/env.sh`
  - `models/Qwen3-32B` (tokenizer + config)
- **Release tools:**
  - `release_tools/verify_assets.py`: sha256 of every file the server loads, against `SHA256SUMS` and `env/ASSETS_PINNED.json`.
  - `qwen-1ch/stage4/voxtral_tools/fetch_voxtral_encoder.py`: cuts the Voxtral audio tower out of the pinned upstream revision.
- **Trainer, for reference** (`ppm/aloe-ft-32-samsep/code/`, everything except backups). The entry point is `train_samsep_vox.py`,
  with its data and lib modules. `vox/run_vox.env`, `vox/run_vox.sbatch` and `vox/stage_init.sbatch` are the launch files. The
  trainer's `model_ppm.py` equals the demo's except for one inert co-train edit (`vmul`, absent from Aloe-only micros). The training
  data stores are not shipped.

No model weights, caches, logs or training data are in `code/`. The only non-code files are the Qwen3-32B tokenizer / config, the
Qwen3-4B config and the demo's small JSON configs. Runtime assets live in `assets/`.

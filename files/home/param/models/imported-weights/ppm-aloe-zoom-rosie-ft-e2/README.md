# PPM Aloe+Zoom cotrain export-253 → Rosie FT, 2 epochs (step 108, final)

Speaker fine-tune of the 32B PPM speech-duplex model **`InternalCan/ppm-aloe-zoom-cotrain` export-253** on Rosie Stamp's conversations
(`InternalCan/rosie-stamp-conversations-v1`), co-trained 1:1 per step with the zoom-strangers-5k text task. This repo holds the
**end of epoch 2 (step 108 of 108, final)**, the checkpoint served live as "Rosie FT · 2 epochs (step 108)". The 1-epoch
export (step 54) is in `InternalCan/ppm-aloe-zoom-rosie-ft-e1`.

## Checkpoint

`export-108/`: `ppm-local-bf16-shards-v1`, **world 6** (`rank0..5.pt` + `rep.pt` + `meta.json` + `COMPLETE`), 76.4 GB, plus
`vgain_theta_fp32.json` (the Voxtral A-gain theta; vadapter / vgain were frozen, so it is export-253's bf16 value, s = 0.0030849).
Loads with the aloe-m1vox-live engine (`live_ppm.build_model`) and the release `serve_release.sh` (which accepts this format).

## Training

| | |
|---|---|
| Base | `InternalCan/ppm-aloe-zoom-cotrain` export-253 (raw Aloe + Zoom cotrain, job 26467) |
| Speech data | `InternalCan/rosie-stamp-conversations-v1`: 212 train calls → 322 chunks (≤ 7,500 frames), 34.4 h; Rosie = channel B (assistant) with her **original** Mimi codes; partner = channel A as raw-level 24 kHz audio into Voxtral (no loudness normalisation) |
| Held out | 4 calls (09bb73ac, 4cc4c9fc, 88e8e0f4, d65a446f; 6 chunks) |
| Voice prompt | 100 % of items: a 3–10 s clip of Rosie's original codes from a **different** train call, text lane blank (PAD), code delay 2 (the `v2_voiceB_only` format) |
| Text co-train | zoom-strangers-5k, 1:1 per step (6 Rosie + 6 zoom micros per step), B text CE only |
| Schedule | 2 epochs = 108 steps (this export = step 108, final); WSD warmup 2 then linear decay |
| Optimiser | **full fine-tune** of the backbone at LR 2e-6 (clip 0.5; fp32 master + AdamW on host); proj 3e-5, audio 6e-5, spec 6e-5; speech tower 4 nanosteps per step at 1.5e-5; 0.6B light decoder 16 nanosteps per step at 6e-5; vadapter + vgain frozen; loss weights text 5, rest 1 |
| Hardware | 6 × H100 80 GB, ~16.6 s / step, peak ~63 GiB / GPU |

Held-out eval (`logs/metrics.jsonl`):

| step | Rosie word CE (norm) | Rosie word top-1 | Rosie cb0 CE | Rosie light CE | zoom word CE (norm) |
|---|---|---|---|---|---|
| 0 (export-253) | 1.6909 | 0.5326 | 0.9431 | 3.2581 | 1.2631 |
| 54 (1 epoch) | 1.6335 | 0.5386 | 0.9202 | 3.1984 | 1.2658 |
| **108 (this export)** | **1.6210** | **0.5423** | **0.9251** | **3.1912** | 1.2639 |

`config/cfg.json` is the resolved run config; `logs/train.log` the full log; `code/` the trainer (`train_rosie_full.py`), the Rosie
data loader (`data_rosie.py`, bit-identical micros to the original data path), the data prep and the launch env.

## Serving

- `serving/CK_LABELS.entry.json`: the label entry for the live server (`prompt: voice_clip_only`, default preset `rosie-d65a446f`).
- `serving/presets-rosie/`: the preset bank used live: 4 **Rosie (clone)** presets (9 s of her original codes from each held-out call,
  blank lane, P = 121; the live prefix is identical to the trainer's for all 322 training clips) + the 12 release Aloe presets;
  `serving/build_rosie_bank.py` rebuilds it. Point `ALOE_PRESET_BANK` at it.
- Always use a Rosie voice clip: the model never saw a prompt without one in this fine-tune.

## How to use it

**1. Download** (needs an InternalCan read token):
```bash
hf download InternalCan/ppm-aloe-zoom-rosie-ft-e2 --local-dir rosie-ft
```
This repo = the 2-epoch export only (`export-108/`). The same checkpoint is also branch **`epoch-2`** of
`InternalCan/ppm-aloe-zoom-rosie-ft-e1` (whose `main` is the 1-epoch `export-54/`).

**2. Serve it live** with the duplex call engine from the release package `InternalCan/aloe-ft-vox17127-nopersona`
(`code/ppm/aloe-m1vox-live/`, the same engine as export-253's live demo; 2 × H100 80 GB, ~55 ms per 80 ms frame):
```bash
source aloe.env                                    # from the release package (venv, Mimi, FA3, Voxtral tower paths)
export ALOE_PRESET_BANK=$PWD/rosie-ft/serving/presets-rosie   # Rosie (clone) voice presets + the Aloe presets
export ALOE_WEIGHTS=$PWD/rosie-ft/export-108       # or export-54
export PPMLIVE_DEFAULTS=unbiased CUDA_VISIBLE_DEVICES=0,1
# add serving/CK_LABELS.entry.json to code/ppm/aloe-m1vox-live/CK_LABELS.json under the absolute ALOE_WEIGHTS path
# (prompt voice_clip_only, default preset rosie-d65a446f), then:
bash code/ppm/aloe-m1vox-live/serve_release.sh     # front proxy on $ALOE_FRONT_PORT; ready when the log shows SMOKE PASS
```
`serve_release.sh` must accept the `ppm-local-bf16-shards-v1` format (rank*.pt + rep.pt + meta.json + COMPLETE): its weights check
must allow `meta.json` in place of `.metadata` (one-line change; `live_ppm.build_model` already loads it).

**3. Talk to it:** open the page, pick a **Rosie (clone)** voice preset (keep the voice clip ON: the model was trained with a Rosie
clip in every prompt), loudness match on or an A gain that puts your mic near -23 LUFS, and start a call.

**4. Load it for training / offline use:** `load_init` in `code/train_rosie_full.py` (from the aloe-pose-duplex trainer family)
reads this export directly and reshards it onto any world size; `vgain_theta_fp32.json` holds the frozen Voxtral gain.

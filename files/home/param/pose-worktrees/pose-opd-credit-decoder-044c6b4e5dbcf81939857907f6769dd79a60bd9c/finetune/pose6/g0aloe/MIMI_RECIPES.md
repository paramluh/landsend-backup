# Mimi recipes: 31101 pose rows vs Aloe pose stores vs the monoplex inference engine (pose6 B2, 10-27 12:35 PT)

**Question.** s2p feeds B's speech cb0 at T into the pose tower. The base 31101 learned `tower_p.s_emb` (zero-init at 17850) on the Mimi codes of its pose rows, while the FT feeds the Aloe stores' Mimi codes. Do the two Mimi recipes match?

**Method.** Read only: on-disk build scripts and metadata, plus two small CPU jobs on slinky-18.
- `mimi_check.py` (job 31139) writes `mimi_check.json`.
- `mimi_check2.py` (job 31141) writes `mimi_check2.json`.

## Verdict
**Nominally the same recipe family (lufs23-local), but not the same encoding.**
- **Identical:** the weights, 24 kHz mono, the -23 LUFS pure-gain target with a 0.995 peak ceiling, and the 250-step (10 s) causal attention window.
- **Different:**
  1. **LUFS scope.** The 31101 pose rows normalise each segment on its own (segments are 120 s or less, median about 7 s). Aloe normalises each whole conversation track (minutes long). On 28% of Aloe B tracks the peak guard leaves the track more than 0.5 dB short of -23.
  2. **Encoder state.** The 31101 pose rows use one independent call per segment, so every segment starts cold. Aloe streams statefully over the whole conversation in 30 s chunks, carrying the KV and padding caches.
  3. **Precision and implementation.** The 31101 pipeline runs at dtype float16 (a pipeline-wide setting; the scope of that dtype is not stated), on transformers 4.57.6 with its own `checkpoint_sliding_window_v1`. Aloe runs fp32 on transformers 5.17.0 with the native sliding window plus the upstream fix.
- **Numbers.** On frames where the speaker is saying words, the cb0 histogram distance between fresh and Aloe is JSD 0.084 bits. The noise floors are 0.002 to 0.003.
  - This is about the size of the old-vs-new recipe change measured on identical clips (0.093).
  - But content differs too (TikTok monologue vs app conversation; different speakers, mics and rooms). With no audio on disk for either source, the histogram cannot separate content from recipe.
- **Silence codes.** Aloe's listening-silence codes are 29% of all Aloe B frames, but 0.4% of npl5 frames (31101's current pose rows). So s_emb has barely trained those rows. That is a data-coverage effect, not a recipe effect.
- **Engine.** It encodes only the voice clip, and with NO loudness normalisation. It also takes its prefill audio from the package's own encoding. Both differ from both training sources. This is an existing behaviour, flagged here and not changed (see the memory rule "lufs23-local = correct; raise other paths").

## Table: recipe per source
| knob | (a) 31101 pose rows: fresh plan 1 (+ npl2, npl3) and npl5 | (b) Aloe pose stores (aloe-pose-ft p0..p7) | (c) engine `aloe-pose-monoplex-infer` | same? |
|---|---|---|---|---|
| **Mimi weights** | `kyutai/mimi`, copied locally to `/opt/multimodal/models/mimi`. `runtime.json` gives model.safetensors sha256 `bac7e850…` and config sha256 `aca6f44b…`. That is the HF blob of `kyutai/mimi@89091b3e`, checked against `~/.cache/huggingface/hub/models--kyutai--mimi`. All 400 sampled releases are identical. | `kyutai/mimi`, `model_revision 89091b3e466e…` (`w00.mimi.json` encoder block; all 18,093 B tracks identical) | `ppm/ppm_io.py:26`: `kyutai/mimi@89091b3e`. Alternatively `MIMI_DIR=/data/home/amartyavarma/models/mimi`, whose safetensors sha256 = `bac7e850…`. | **SAME** |
| **sample rate** | 24 kHz mono. `audio_sr=24000` on all 7,523 checked segments. `FORMAT.md:39` | 24 kHz mono: "ffmpeg f32le mono 24k (atrim -> apad -> resample)" | `ppm_io.read_wav24` (`:132-140`): channel mean, then `scipy resample_poly` to 24 kHz | **SAME** rate (resamplers differ) |
| **loudness** | -23 LUFS pure gain, 0.995 peak ceiling, **per independent segment** (`normalization_scope=independent_training_segment`, all 7,523 segments). Achieved: p5 -23.4 (plan 1) / -23.15 (npl5), p50 -23.0. Peak-limited: 6% (plan 1), 5% (npl5). | -23 LUFS, "pyloudnorm BS.1770-4 integrated, pure gain, peak-guard 0.995", **per whole conversation track**. Peak-limited: 6,054 of 18,093 tracks. Achieved: p5 -26.2, p25 -23.8, p50 -23.0. 28% of tracks are more than 0.5 dB short. | **NONE.** `ppm_io.Mimi.encode` (`:127-129`) passes the raw wav straight to Mimi. | (a) vs (b): same target, **different scope**. (c): **DIFFERENT** |
| **attention / context** | "corrected checkpoint-local causal attention, 250 steps / 10 s per layer" (`new-pose-llm-data/README.md:55`; `attention_policy=checkpoint_sliding_window_v1` on all 7,523 segments) | `sliding_window_encoder_frames 250`, `sliding_window_seconds 10.0`, "automatic_model_internal_sliding_attention", transformers 5.17.0 (the fix first shipped in 5.3.0, upstream commit 7c0a3be) | transformers 5.13.0 (`requirements.txt:5`, newer than the 5.3.0 fix), config `sliding_window` 250. The clip is cropped to 125 frames or fewer (`infer.py:169`), so the window never binds. | **SAME** 10 s window (different implementations) |
| **encoder state / call** | `audio_encode=one_independent_call` per segment; "normalization/model state reset per unit" (`README.md:53`); `mimi_max_encode_s` 120; `tokens_version v5-independent-120s-lufs23-local-v1`. Every segment starts **cold**. | `MimiModel.encode(use_streaming=True)` in "fixed frame-aligned max 30 s / 375 output frames" chunks, `state_carried [encoder_past_key_values, padding_cache]`. One **warm** stream per conversation track. | One non-streaming `encode(x, num_quantizers=32)` call per voice clip (`ppm_io.py:127-129`, `infer.py:66-71`) | **DIFFERENT** |
| **dtype** | `runtime.json config.dtype float16` (pipeline-wide; scope not stated) | `fp32` (TF32 matmul off) | fp32 (`from_pretrained` default, `ppm_io.py:116`) | possibly **DIFFERENT** |
| **recipe tag** | `mimi_audio_recipe = audio_policy = lufs23-local-v1` (400 of 400 runtime.json files; 7,523 of 7,523 segments) | `encoder_version mimi-stream-v5` ("locked", `DATA/README.md:27`) | none | names differ |
| **row builder** | `fresh/fresh_pose.py:93-95` copies `audio.mimi_codes` verbatim. It flags only `audio_sr` (`:125-126`) and never reads `segment_metadata_json` (`COLS`, `:61`), so there is **no recipe check**. | `aloe-pose-ft/code/build_poseB_rows.py:110` copies `w00.<SPK>.codes.npy` verbatim (no re-encode) | n/a | |

Coverage of (a), all from on-disk files:
- **Plan 1:** a census of 400 release `runtime.json` files from `/data/home/amartyavarma/data/new-pose-llm-data`, plus 1,018 segments from 11 parquet files.
- **npl5 (delivery 5):** 6,505 segments from 60 of the 744 parquet files in the `pose5/npl5-weka` window.
- **npl2 and npl3:** these exist only on the 31101 nodes' /scratch, so they are **not checked**. The user said "same format", and the builder does not check the recipe.
- s_emb is zero-initialised at 17850, so it learns only on the npl3 tail rows and then npl5.

## Numeric spot check: cb0 unigram histograms (2,048 bins; JSD in bits)
| comparison | all frames | word-active frames |
|---|---|---|
| plan 1 vs npl5 (same recipe, same domain) | 0.011 | **0.0019** |
| noise floor: split halves within npl5 / plan 1 | 0.017 (plan 1) | 0.0021 (npl5) |
| noise floor: split halves within Aloe B | 0.0048 | 0.0033 |
| **fresh (plan 1 / npl5) vs Aloe B** | 0.286 / 0.281 | **0.084 / 0.085** (TV 0.245 / 0.248) |
| calibration: the same 283 TikTok clips, old TikTok encode vs lufs23-local (recipe-only change; `fresh/mimichk`) | 0.093 (0.295 on frames at or after 10 s) | n/a |

Frames per source:
- all frames: 147k (plan 1), 1.17M (npl5), 1.87M (Aloe B; 400 random items).
- word-active frames: 1.32M (plan 1), 0.94M (npl5), 0.94M (Aloe).

Word-active fraction: 0.80 for plan 1 and npl5, but **0.47 for Aloe B**. Aloe's B track spends half its time listening.

Aloe's top-5 all-frame codes are 2029, 1316, 1926, 948 and 175:
- They make up **29.1%** of all Aloe B frames, but only 0.19% of its word-active frames. So they are listening silence.
- They make up **0.38%** of all npl5 frames, and 0.0% of npl5 word-active frames.

The top-5 word-active codes do not overlap between the sources: fresh {1174, 1532, 1940, 1087, 13} vs Aloe {1193, 1017, 569, 1633, 1345}. Each top-5 holds only about 1.2-1.7% of frames, so the distributions are flat.

`fresh/mimichk/compare_out.txt` (same clips) gives cb0 agreement between the old encode and lufs23-local of 0.89 for frames under 10 s and 0.25 for frames at or after 10 s. This shows that the attention-context part of a recipe changes cb0 far more than the gain part.

## What this means for s2p in the FT (no decision taken here)
- **Weights, rate, gain target and window match.** The Aloe stores are lufs23-local-family codes, not the old "wrong" encode. Per the memory rule, nothing should be re-encoded.
- **Residual mismatches:**
  - LUFS scope (track vs segment, with peak-guard shortfall on 28% of tracks).
  - Warm streaming vs cold per-segment state.
  - fp32 vs fp16.
  - Their combined effect on cb0 is unknown. It is bounded above by the 0.084-bit word-active JSD, which also contains the content shift.
- **Listening silence (about 29% of Aloe B frames) uses codes that 31101's pose rows almost never contain.** s_emb rows for those codes stay near their zero init, so the pose tower gets roughly no s2p signal on listening frames until the FT trains them.
- **Engine voice clips skip -23 LUFS.** This is a pre-existing inference difference from both training encodings. It is worth raising with the user: the memory rule says to raise it, never to change it silently.
- A decisive same-audio comparison (re-encoding one Aloe track both ways) needs the Aloe WAVs. They are not on disk (`DATA/NOTE_local.md`: no FLAC), and downloads are not allowed without an OK.

Files:
- `/data/home/amartyavarma/ppm/pose6/g0aloe/mimi_check.py` / `.json` / `.sbatch`
- `/data/home/amartyavarma/ppm/pose6/g0aloe/mimi_check2.py` / `.json` / `.sbatch`
- logs: `logs/mimichk-31139.log`, `logs/mimichk2-31141.log`

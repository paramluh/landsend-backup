# posefix COMPILE (10-26, user via main: "ultra compile everything")

Built on stage-exact, the trainer-exact light decoder that has been live since 07:13 UTC.

## Setup

- `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)` wraps every captured per-frame function. The live engine has: text, audio, audio_full, pose, pose_full, pose_full_cfg, seg0/1, front (Voxtral), mdec (Mimi decode).
- The compiled functions run INSIDE the engine's own CUDA graphs, captured as before.
- Inductor settings:
  - `fallback_random=True` and `emulate_precision_casts=True`;
  - `max_autotune_gemm_backends=ATEN` (cuBLAS, as eager).
- Two ops are wrapped as opaque `torch.library` custom ops so dynamo traces each function whole. Same kernel, same arguments:
  - the generator draw: `ppmlive::multinomial1` (`fastppm::` in fast_ppm), because dynamo cannot take a `torch.Generator` argument;
  - the FA3 KV-cache attention: `ppmlive::fa3_kvcache`. FA3 has no fake impl, and its schema hides the k/v cache writes, so the custom op declares `mutates_args=(k_cache, v_cache)`.
- With compile OFF, these wrappers are bit-identical to stage-exact:
  - live free-generation lane/codes/pose all equal;
  - mono all equal.
- The 0.6B decoder loops stay UNCOMPILED inside the compiled graphs (`torch.compiler.disable`; still CUDA-graphed). Default `PPMLIVE_COMPILE_DEC=0` / `FASTPPM_COMPILE_DEC=0`. Measured on the speech decoder:
  - graphed 26.78 → 26.49 ms (−1%);
  - logits top-1 0.990, max|Δ| 0.25, CE 2.2582 → 2.2582;
  - ~13 min cold compile per decoder graph (×3 graphs).
- Toggles:
  - `PPMLIVE_COMPILE=0` / `FASTPPM_COMPILE=0` = the eager (uncompiled, graphed) engine;
  - `PPMLIVE_COMPILE_SKIP=<graph names>` leaves named graphs uncompiled.
  - A graph that fails to compile falls back to uncompiled and is logged in `COMPILE_STATUS`.
- Inductor cache: `/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/compile/inductor-cache`. It is shared across nodes and is 413 MB now.

## Duplex live engine (export-498 dcp, slinky-18 GPUs 0-1)

Run on 2 held-out convs × 300 frames, teacher-forced, graphed.

**Per-frame ms (graphed free generation):**

| Engine | p50 ms | p95 ms |
|---|---|---|
| Uncompiled | 64.15 | 64.25 |
| Compiled | 63.39 | 63.48 |
| stage-exact (live now) | 63.99 | — |

Compiled is 1.2% faster than uncompiled.

**Compiled vs uncompiled logits:**

| Head | top-1 agree | TV | max\|Δ\| | CE uncompiled → compiled |
|---|---|---|---|---|
| Text | 0.998 | 0.0028 | 1.375 | 0.3442 → 0.3443 |
| Speech cb0 | 0.993 | 0.0056 | 0.17 | 1.0015 → 1.0030 |
| Pose cb0 | 0.990 | 0.0077 | 0.25 | 1.0287 → 1.0286 |

**Other results:**

- Free generation, same seed: lane 86% equal, but it diverges from frame 9. Compiled is deterministic across runs.
- Capture with compile: 299 s with a cold-ish cache (uncompiled: 49 s). A TRITON-GEMM variant gave the same ms and the same cb0 diffs, and took 487 s.

## Monoplex playground fast_ppm (export-357)

Run on 2 convs × 200 frames.

**Per-frame ms:**

| Engine | p50 ms |
|---|---|
| Uncompiled | 60.83 |
| Compiled | 60.26 |
| stage-exact | 60.78 |

Compiled is 0.9% faster than uncompiled.

**Compiled vs uncompiled logits:**

| Head | top-1 agree | TV | CE uncompiled → compiled |
|---|---|---|---|
| Text | 0.992 | 0.0029 | 0.4374 → 0.4368 |
| Speech cb0 | 0.997 | 0.0044 | 0.8055 → 0.8050 |
| Pose cb0 | 0.995 | 0.0036 | 0.6749 → 0.6769 |

**Other results:**

- Free generation: lane 85%, codes 55%, pose 47% equal. Compiled is deterministic.
- Capture with compile: 287 s (uncompiled: 6 s).

## Verdict

- Compile works on every graph. It is ~1% faster and not bit-identical.
- The logit drift is comparable in size to the pure-bf16 decoder departure just fixed. TV is 0.003-0.008 on the cb0/text heads, and CE is unchanged to about ±0.002.
- Server start costs about +4-5 min on a cold cache; it is faster once the shared cache is warm.

## Deploy (idle-only, via aloedemo)

| Target | Source | Destination |
|---|---|---|
| live ×2 instances + duplex playground worker | stage-compile/live_ppm_pose.py | LIVE/live_ppm_pose.py |
| same | stage-compile/light_exact_ppm.py (adds a compile-safe device guard; eager identical) | LIVE/light_exact_ppm.py |
| monoplex playground | stage-compile/mono/fast_ppm.py | aloe-pose-monoplex-infer/ppm/fast_ppm.py |
| same | stage-compile/mono/light_exact_ppm.py | aloe-pose-monoplex-infer/ppm/light_exact_ppm.py |

- Rollback is the stage-exact files, which are live now. Alternatively, set `PPMLIVE_COMPILE=0` / `FASTPPM_COMPILE=0` in the env and restart.
- Raw results:
  - test_compile_30607d.json: duplex, ATEN;
  - test_compile_triton_gemm.json;
  - test_compile_mono_30607.json;
  - light_speech_decoder_compile.txt.

# Cached inference

`cached_universal.py`, config `cached_universal.yaml`: `inference/pose_universal.py`'s sampler run frame → band → substep (same `t`, windows and trajectories) with the left context read from per-layer K/V instead of being run through the backbone again. A step runs only the moving frame's N = 120 tokens. H = 4 history, P = 5 present, m = 10 substeps per band, 50 steps per frame (`num_steps = substeps × num_present` is checked). Requires `type: universal_wide`.

## What is cached

- `inference/pose_universal.py` stores **frame latents** (`clean`, `front`) and rebuilds every window from them. Caching stores the **per-layer K/V** those latents become.
- Attention rule (`TransformerLayer._fixed_attention`):
  - sink: itself;
  - history frame: sink + itself, q and k rotated at the anchor slot H, so its hidden states do not depend on its slot;
  - present slot i: sink + the H frames before it + itself, at true slots.
- The moving frame `k` at band `b` sits at slot `1 + H + b` and reads only `[sink | slots b+1 .. H+b | self]`: H + 2 = 6 frames, 720 keys. Slots to its right are never its keys.
- Bands run P−1 → 0. Band `b`, substep `s`: `t = (b+1)/P + s·dt`, `dt = −1/(P·m) = −0.02`, fed as `t · 1000` in fp32.

## Entries

- **Sink** (slot 0): GT frame 0, `t = 0`, no pose, attends itself.
- **History**: the hidden states are context-free, but present queries read a history frame's K at the slot it currently occupies, and that slot moves one left at every commit. K rotated at the anchor would be right only at slot H. So the commit stores K rotated at every slot 1..H the frame will occupy. V has no RoPE: the same V goes into all H entries.
- **Present `(frame, band, s)`**: the moving frame's own K/V at that step, rotated at its true slot `1 + H + band`. Frame `frame + j` reads it at band `band + j`, same `s`: same slot, same `t`, same left context. Band P−1's K/V is never read and goes to scratch.
- **Pads**: a startup hole (frame < 1) is the learned pad token at `t = 0`, no pose, no audio.
  - History pads: one pass `[sink | pad]` at the anchor, with K stored at slots 1..H.
  - Present pad at present slot `j`: attends sink + history pads at slots `j+1..H` + present pads `0..j−1`. Computed in order, `j = 0..P−2`.
  - Read in place of missing frames; older real frames never slide into the holes.

## Pool

One pool for K and one for V, each `[30 layers, 230 entries, 120 tokens, 30 heads, 128]` bf16 (5.9 GiB each, 11.8 GiB total). Every entry holds K already rotated at the slot it is read from.

| entries | count | content |
|---|---|---|
| 0 | 1 | scratch: K/V of band P−1 steps, so the step has one shape |
| 1 | 1 | sink |
| 2–5 | H | history pads, slots 1..4 |
| 6–9 | P−1 | present pads, present slots 0..3 |
| 10–29 | (H+1)·H | history ring: frame `f` at slot `i` → `10 + 4·(f mod 5) + i − 1` |
| 30–229 | P·(P−1)·m | present ring: `(f, band, s)` → `30 + ((f mod 5)·4 + band)·10 + s` |

- Frame `k` reads history frames `k−H..k−1` and present frames `k−P+1..k−1`. The rings overwrite `f − (H+1)` and `f − P` with no explicit drops.
- A frame's own entries are never in its context, so the pool writes come after the pass.

## Per clip

`start_clip`:
1. Clear the audio cache.
2. Sink pass.
3. History-pad pass, stored at all H slots.
4. Present-pad passes `j = 0..P−2`.

The graphs are captured on the first clip. `schedule(L)` then builds every step's indices on the GPU.

One noise draw `[C, L, h, w]` per clip, seed = `seed + n`; frame `k` starts from `noise[:, k]`, with its GT pose.

## Batching

- `batch_size` (default 1) clips run together: every buffer has a leading clip dim, and clip `b` owns pool entries `b · 230 + e` (the pool is `batch_size` × 11.8 GiB).
- The step schedule is shared; only the audio is per clip (each clip slides and encodes its own table into its row of `buf_audio`).
- A batch runs to its longest clip. A shorter clip repeats its last frame's audio past its end and keeps only its own frames, which never read later ones. A short last batch is filled with copies of its last clip.

## One cached step

Only the moving frame's 120 tokens:
1. **Initial**: patch embedding of the state, plus the pose embedding, plus the time embedding, giving the AdaLN modulation.
2. **Each of 30 blocks**:
   - AdaLN, then q/k/v on 120 tokens, `norm_q`/`norm_k`, and RoPE on q and its own k at slot `1 + H + b`;
   - FA3 `120 × 720` against `[sink | H context entries | self]`, gathered from the pool by index;
   - o;
   - on the 8 audio blocks, cross-attention `120 × 5` (K = 4 codes + null);
   - FFN.
3. **Final** head (fp32 autocast) produces the velocity `[C, h, w]`.
4. Write its per-layer K/V to `(frame, band, s)` (scratch at band P−1), then update the state in place with Euler.

Audio guidance (off in the yaml: scale 1.0, 0 steps). On a frame's first `audio_guidance_steps` steps, a second pass with `no_audio_tokens` reads the same cache and stores nothing. The step then uses `v = v_noaudio + w·(v_audio − v_noaudio)`.

## Commit

After band 0's last substep the state is the finished frame. One pass with:
- `t = 0`;
- q and its own k at the anchor (slot H);
- context = sink only;
- pose on, no audio, no head.

Per-layer K, rotated at slots 1..H, and V go into the frame's H history-ring entries. The finished latent is appended to the output.

## Audio

- The code table slides as in training (`sliding_code_table`): 750 codes (60 s, 375 latent frames).
  - A window whose last real frame is `f_last` reads codes from `2a`, with `a = max(0, f_last − 374)`, and frame indices shifted down by `a`.
  - `f_last` is the wavefront's, `min(k + P − 1 − b, L − 1)`, so the trajectories equal `inference/pose_universal.py`'s for clips of any length.
- Mimi latents and the audio encoder run once per table start `a` (one encode for the batch's tables), not per forward. The result is cached per `a`; starts below `a − P` are dropped.
  - Clips up to 375 frames: once per clip.
  - Longer clips: about once per frame after that.
- Before each band (`set_audio`), the moving frame's 4 tokens (index = frame − `a`) plus the null token are gathered from the encoded table into the static `buf_audio`. The band's substeps and no-audio passes reuse it.

## bf16 weight casts

The audio blocks and the pose patch embedding are fp32, as in training. Autocast would cast their Linear/Conv3d weights to bf16 on every call; they are cast once at init instead, giving the same bf16 operands. Their RMSNorm weights stay fp32, as autocast leaves them.

## Compile + CUDA graph

- `_moving`, `_moving_noaudio` and `_commit` are pure functions of static buffers: state, pose, int indices, store indices and encoded audio. Each returns the velocity and the K/V to store, and each is `torch.compile(dynamic=False)`.
- CUDA graphs, captured with their pool writes:
  - step = pass + K/V write + Euler update;
  - commit = pass + H-entry write;
  - guided step, only when guidance is on.
- Capture: 3 warm-up calls on a side stream (the first compiles), then capture.
- GPU-side schedule per clip:
  - one row per step in sampling order, `[ctx (H+1), write, band, s]`;
  - one row per frame with its H history entries;
  - one row per band, each clip's `frame − a`, indexed by `set_audio` (`audio_schedule`).
- The loop is device copies (noise, pose, schedule row into static buffers), the audio encode and window gather (eager) and graph replays. There is no host sync except the progress print every 100 frames.
- Whole-step compile + one CUDA graph is the fastest step in `benchmark.md`.

## Triton config (reproducibility)

- Inductor's default times each Triton kernel's candidate configs at runtime and keeps the fastest. In the norm reductions the config changes the summation order, so outputs differed per process and per node from the first step.
- `cached_universal.py` patches `CachingAutotuner.autotune_to_one_config` to keep the first candidate config (saved to the cache). It also sets `torch.compiler.config.cache_key_tag = 'first-config'`, so caches with timing-picked configs are not read.
- Result (`docs/corruption/conclusions.md`, checks in `/home/vansh/eval/repro/`):
  - fresh runs on r4 (two GPUs), r3 and r2 are bit-identical at 365 ms/frame;
  - production `main` (clip 03, 30 frames, cold cache) is bit-identical on r4, r0 (two GPUs) and r3, at 367–380 ms/frame.
- Eager and compiled never match each other. Noise depends on clip length, so a truncated clip gets different noise for the same seed.

## Checks

Accuracy (aloe-h4 step 4627, 40 frames, vs the since-removed uncached `universal.py`). It is not bit-identical to the full window, because the GEMMs see 120 rows instead of the window's.
- Uncompiled + graph: per-frame rel-L2 0.4% at frame 1, plateau ~1.2% (max 1.6%).
- Compiled + graph: 0.45% at frame 1, rising to 2.2% at frame 39.
- Generated vs ground truth: ~45%.
- The graph matches eager bit for bit, both uncompiled and compiled.

Speed (per GPU; RTF = sampling time / video duration, one latent frame = 4 video frames = 0.16 s at 25 fps):
- Uncompiled + graph: 14.25 ms/step, 713 ms/frame, RTF 4.45.
- Compiled + graph: 7.15 ms/pass, 365 ms/frame (50 steps + commit), 16,780 query tokens/s, RTF 2.28.
- `universal.py` uncached (since removed): 2.89 s/frame.
- Production: ~366 ms/frame. A 5-min clip (1,874 latent frames) takes ≈ 11.5 min + ~3 min decode, ~80 GB per GPU.

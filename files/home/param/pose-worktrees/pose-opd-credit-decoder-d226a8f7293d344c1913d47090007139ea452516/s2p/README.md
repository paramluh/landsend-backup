# pose-s2p: speech now conditions pose (branch)

The user asked on 10-27 at 05:15 PT for "speech now conditions pose also" (board t1027-24; design t1027-21).

At each frame T, the pose tower's input gains one new term: `s_emb` applied to channel B's speech (Mimi) codes at the **same** position T. Those are the codes of the very frame whose pose is predicted at T. The timing is unchanged: pose frame f and speech frame f still leave in the same step. Inside a step, the order becomes text -> speech -> pose.

The change sits behind the flag `PPM_S2P`:

| value | meaning |
|---|---|
| `0` or unset | off |
| `1` or `full` | all 32 speech codebooks |
| `cb0` | speech cb0 only |

With the flag off, no module is built, no key exists, and every line of today's maths runs unchanged. Tests are in `TESTS.md`.

**Status (10-27 06:35 PT).** Tests (a), (b) and (c) PASS.
- **(a):** in the deployed layout, flag off, full and cb0 at zero init are bit-identical to the original deployed engine, eager and with CUDA graphs. The old-vs-old floor is 0.
- **(d) latency per frame:** today 61.0 ms, full 96.3 ms (over the 80 ms live budget), cb0 66.3 ms.
- **(e):** the fine-tune is a dry config only and is NOT launched.

## Files

| file | from | what changed |
|---|---|---|
| `code/model_ppm.py` | `aloe-pose-rosie/code` (= mono8020 + the Rosie fixes, all env-gated) | adds `s2p_mode`, `S2PEmb`, `Tower.forward(sdep=)`, `PPM(s2p=)`, `init_rule`, `build(s2p=)`. Diff: `code_model_ppm.diff` |
| `code/train_aloe_pose.py` | same | adds the `PPM_S2P` flag, `CFG["s2p"]` (only when on), `build(..., s2p=)`, and `load_init` (zero-fills a missing `s_emb`). Diff: `code_train_aloe_pose.diff` |
| `infer/ppm/fast_ppm.py` | the DEPLOYED engine `aloe-pose-monoplex-infer/ppm/fast_ppm.py` (genweb worker.py imports this one, not `ppm/fastinfer`) | see "Inference" below. Diff: `infer_fast_ppm.diff` |

No other file is changed. The data code is untouched: every pose micro already carries B's speech codes in `b["depth"][1]`.

## Port to code-811 (pose5): the exact model and trainer change

### 1. `model_ppm.py`, after `DEPTH, CBS = ...`

Add `s2p_mode(v)`:
- `0`, empty, `off`, `none` or `false` -> `None`;
- `1`, `full`, `on` or `true` -> `"full"`;
- `cb0` -> `"cb0"`;
- anything else -> assert.

Add the class `S2PEmb(nn.Module)`:
- `weight` = `nn.Parameter(empty((32 if full else 1) * 2048, H_tower))`;
- `forward(depth)` = `H2.code_feedback(depth if full else depth[..., :1], weight)`. This is the same sum-of-rows table as `CodeTable`, with -1 -> 0.

### 2. `Tower.forward(..., info=None, sdep=None)`

The input sum becomes:
```python
e = e + ce_.to(e.dtype) + cnd.to(e.dtype)
if sdep is not None and hasattr(self, "s_emb"):
    e = e + self.s_emb(sdep).to(e.dtype)      # NOT shifted: codes at T
e = e[None]
```
With the flag off, this is the old `(e + ce_ + cnd)[None]` with the same op order.

### 3. `PPM.__init__(..., s2p=None)`

```python
self.s2p = s2p_mode(s2p)
if self.s2p:
    self.tower_p.s_emb = S2PEmb(c4.hidden_size, self.s2p)
```
- The key is `tower_p.s_emb.weight`.
- `group_of` puts it in group `tower_p` (same LR and clip).
- It sits inside the `tower_p` FSDP unit.

In `forward`, in the `kind == "pose"` branch:
```python
self.tower_p(..., b["pcb0"], b["pcb0_w"], sdep=b["depth"][1] if self.s2p else None)
```

### 4. `init_rule` and `build`

- `init_rule`: `tower_p.s_emb.weight` -> `("zeros", 0.0)`, placed before the generic tower rules.
- `build(..., s2p=None)` passes it to `PPM`.

### 5. Trainer

```python
S2P = M.s2p_mode(os.environ.get("PPM_S2P", "0"))
```
- If set, add `CFG["s2p"] = S2P`. With the flag off, the resolved config is unchanged.
- Pass `M.build(..., s2p=S2P)`.
- Assert `bool(S2P) == hasattr(model.tower_p, "s_emb")` and that the pose path is on.
- In the init loader, skip `tower_p.s_emb.weight` when the checkpoint lacks it, then `zero_()` it. Every other missing key still asserts.

### Rationale

- **Sampled codes, not the speech latent.** `hv_s[T]` is a deterministic function of inputs the pose tower already sees: the word token and the history. The random draw (the codes) is the only new information.
- **Same position T, unshifted.** `depth[1][T]` holds the Mimi codes of frame T - CD. That is the same frame as pose label `pdepth[T, 0]`. In training it is teacher-forced, and the real codes are already in every pose micro.
- **The zero table is the output projection.** A code table is a linear map of the one-hot codes. Zero-initialising it makes step 0 bit-identical to today's model, and every used row still gets a gradient from step 0. This is test (c): the `s_emb` grad norm is 0.199 from the pose loss alone. A random table followed by a zero projection would give the table exactly zero gradient at step 0.
- **Positions without speech codes get 0.** These are the system block outside the voice clip, conversation frames 0 and 1, and text rows. The voice clip's codes do reach `s_emb` at clip positions, which have no pose label; inference prefill does the same.
- **No new loss, no new optimizer group, no data change.**

## Inference (`infer/ppm/fast_ppm.py`)

How the flag is chosen:
- `build_model` follows the checkpoint: the term is on iff `tower_p.s_emb.weight` exists, and the mode comes from its row count.
- Otherwise `PPM_S2P` decides, and a missing weight is zero-filled.
- `FastPPM(..., s2p=None|False|S2PEmb)` defaults to following the model.

What the branch changes, with the flag on:
- **Frame order.** `step()` runs text -> speech -> forced speech codes -> codes copied to the pose GPU -> pose -> forced pose -> trunk.
  - `full`: pose waits for the whole speech branch (tower, cb0 and the 31-codebook decoder).
  - `cb0`: speech is split into two CUDA graphs, `speech_head_full` (tower and cb0 draw) and `speech_light_full` (the light decoder). Pose starts right after the cb0 draw and runs concurrently with the speech light decoder. `gen_a` is the same generator and is drawn in the same order.
- **Prefill.** The pose tower prefill adds `s_emb(depth[1])`. Uncond rows receive `silence_depth`'s codes, as today.
- **Uncond pose-CFG rows.** `s_emb` reads the Mimi codes of digital silence, `sil_t[T - P - CD]`, where B has codes at T, and -1 elsewhere. This is the same rule `_seg` applies to the trunk's uncond input.
- **Audio CFG.** This engine has no audio-CFG rows, so there is nothing to handle.

With the flag off, `step()` is today's code (only re-indented under `else`), and the graphs are the same.

**Duplex engine** (`aloe-posedup-live/live_ppm_pose.py`, out of scope) would need the same three pieces:
- the `s_emb` term in its pose step and its prefill;
- the speech-before-pose order in its frame loop: today pose and speech run concurrently there too;
- the silence rule for its pose-CFG uncond rows.

There is one duplex-specific choice: `s_emb` must read B's own speech at T only, never A's.

## Fine-tune

The dry launch config is `s2p_mono8020.env` with `s2p_mono8020_run.sh` and `s2p_mono8020.sbatch`. It is NOT submitted; see `TESTS.md` (e).

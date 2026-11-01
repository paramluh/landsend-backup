# History corruption

An autoregressive model conditions every new frame on its own previous output. A small error in that history, such as a colour shift or a mosaic patch in the background, is copied into the next frame, which then becomes history itself. Over minutes the error accumulates.

History corruption trains the model not to copy errors from history. In some training windows the history is deliberately damaged while the sink (frame 0) and the pose stay clean. The model has to reproduce the clean target anyway, so it learns to check its history against the sink instead of copying it.

**Code:** `HistoryCorruption` in `models/corruption.py`, with the mosaic generator in `models/mosaic.py`.
**Config:** the `[history_corruption]` block of the experiment toml.
**Inference:** `production/adaptive_sampling.py` uses the same class, with `color_p` and `mosaic_p` rescaled to sum to 1; see `production/adaptive_sampling.md`.

## Method

### Scope

- A window is `[sink | num_history history | num_present present]` latent frames. Corruption runs on training windows only, on the clean latents before noising.
- Only real history slots (original frame index ≥ 1) get `d`. The sink, padding slots, present frames and pose latents are never touched, so the target is unchanged.
- **Units:** `d` is in per-channel stds of the window's real history latents, each channel pooled over its real history slots and cells. The same setting is the same relative change on any clip.

### One draw per window

- A uniform `u` per window picks the kind: mosaic if `u < mosaic_p`, colour if `u < mosaic_p + color_p`, else clean. Windows without a real history slot stay clean.
- Everything else is drawn once per window and shared by its slots: direction, sign, amplitudes, field, envelope, coverage, slot profile and n. The corrupted history therefore agrees with itself, and only the sink shows it is wrong, as in real drift.

| window | `d` |
|---|---|
| colour | `colour · (a_person + mask · a_background)` |
| mosaic | `pose_background_mask · envelope · (a · field + mosaic_dc · a · MEAN_DIRECTION)`, no colour term |
| clean | none |

### Colour

- **`colour`:** a 48-channel direction, normalised to unit channel-RMS, with a uniformly random sign. It is applied as a per-channel constant (DC) offset with no texture; its only spatial shape comes from `mask`. `directions` gives the probabilities of the three kinds:
  - DARK: the latent change of a pixel gain below 1 (positive = darker);
  - DRIFT: a measured rollout background drift;
  - random: a fresh normalised randn.
- **Person / background split:**
  - `a_person` ~ log-uniform(`person_color`) is applied over the whole frame, person included.
  - `a_background` ~ log-uniform(`background_color`), drawn independently, is added on the background only, weighted by `mask`.
- **`mask`** (`background_mask`): a soft background weight from the clean frame and its pose latent.
  - Wall test: a cell is far from the wall if its colour is more than `mask_tau` per-channel spatial stds from the frame's top-corner mean (top 4 rows, 2 leftmost and 2 rightmost columns).
  - Pose skeleton: pose cells whose deviation from the frame's median pose exceeds the median deviation by `mask_tau_pose`, grown by one cell.
  - Person = far cells ∪ skeleton, grown by one cell. `mask` = 1 − person, feathered 3×3.
  - The wall test is conservative: it covers only part of the true background and does not reach the person.

### Mosaic

The mosaic imitates the main background artifact of long rollouts: saturated, many-coloured, irregular blobs that start at the top of the background and on the side walls.

- **`field`:** broadband, unit channel-RMS.
  - It is the sum of n ~ U{`N_DIR`} Gaussian scalar fields. Each field has the radial band spectrum fitted to real mosaics (`BAND_POWER` over `PERIOD_EDGES`, no DC).
  - Each field lies along its own channel direction, drawn from N(0, bank covariance + unexplained variance spread isotropically).
  - The bank (`BANK`, `BANK_VAR`) is the top 28 PCA directions of the per-cell non-DC deviation of 13 real mosaics, and holds 0.90 of their variance.
- **Amplitude:** `a` ~ log-uniform(`mosaic`) is the non-DC channel RMS of `field`.
- **`MEAN_DIRECTION`:** the 13 mosaics' mean DC direction, scaled to unit channel-RMS with a fixed sign (pairwise cos 0.82–0.99, same sign). It is added with DC `mosaic_dc · a`.
- **`envelope`:** top-biased smooth support.
  - The score is s = −slope · row / (h − 1) + a smooth random field, with slope ~ U(1.5, 4.0).
  - The highest-s cells are switched on until they cover a fraction `coverage` ~ U(`mosaic_coverage`) of the mask weight, then feathered 3×3.
- **`pose_background_mask`:** 1 − the pose hull, grown by one cell and feathered 3×3, with no colour test.
  - The hull is built from each skeleton row's horizontal span. The head span is carried down over the face rows, and a hair cap of max(3, 0.75 × head width) rows sits above the head.
  - It covers most of the true background, with a small leak at the person's edge.

### Slot profiles

- `slots` = [full, leading, trailing] probabilities pick which real slots get `d`: all of them, the newest n, or the oldest n, with n ~ U{1, …, num_history − 1}.
- Slot order is oldest → newest. n counts over the real slots, and n ≥ the number of real slots gives all of them.
- The leading and trailing profiles match corrupt-and-cache at inference, where the newest few frames are corrupted, or only the oldest few are left corrupted.
- `per_slot_mask`: each slot's `mask`, and the support of its envelope, come from its own clean frame and pose. The envelope's random field, slope, coverage and all amplitudes stay shared. This matches inference, where each frame is corrupted at its own commit.
- Without `slots`, every real slot gets `d`, with no extra random draw. Without `per_slot_mask`, all slots use the newest history frame's mask and pose.

### History dropout and padding

- `history_dropout[d−1]` is the probability of dropping the d oldest history frames of a window, in the data loader. Dropped frames and clip-start frames become padding, which is never corrupted.
- Padding always sits on the oldest slots, so the real slots are the newest contiguous run, and the slot profile counts over them.

### Strengths

- `a_person`, `a_background` and `a` are log-uniform over their ranges: one range each, with no strength bands. `coverage` is uniform.
- Five log-spaced points from min to max are the 0/25/50/75/100% quantiles of a log-uniform draw, so each gap between them holds 25% of draws.

### Calibration

- A window of clean context latents plus the history latents with `d` on every slot (per-slot masks) is decoded, and the newest history frame is compared with the decode of the same window without `d`.
- Metrics, as in [AutoBench](autobench.md):
  - background ΔE (CIE76, M1);
  - background texture log-ratio (M2);
  - person ΔE on the eroded pose hull.
- Scripts: `/home/vansh/eval/drift_vs_corruption/weak_calib.py` (medians over windows) and `draws_viz.py` (single draws and the draw grids).
- The class also has a checkerboard paradigm (Nyquist grid, selected when the config has `grid`); the final run does not use it.

## Final run: `aloe-h4-complete-corruption`

**Config:** `experiments/universal/aloe-h4-complete-corruption.toml`. Window `[sink | 4 history | 5 present]`.

**Shares:** mosaic 23.3%, colour 46.7%, clean 30%.

| key | value | meaning |
|---|---|---|
| `mosaic_p` | 0.233 | mosaic windows |
| `color_p` | 0.467 | colour windows; the remaining 0.3 are clean |
| `directions` | [0.4, 0.4, 0.2] | DARK, DRIFT, random |
| `person_color` | [0.018, 0.2] | `a_person` range |
| `background_color` | [0.059, 1.2] | `a_background` range |
| `mask_tau` | 0.8 | wall test threshold |
| `mask_tau_pose` | 0.3 | pose skeleton threshold |
| `mosaic` | [0.35, 1.8] | `a` range |
| `mosaic_dc` | 0.8 | mosaic DC = 0.8 · `a` |
| `mosaic_coverage` | [0.7, 1.0] | envelope coverage range |
| `slots` | [0.4, 0.3, 0.3] | full, leading, trailing; n ~ U{1, 2, 3} |
| `per_slot_mask` | true | per-slot masks |
| `history_dropout` | [0.1, 0.1, 0, 0] | drop the 1 or 2 oldest history frames, 10% each |

| amplitude | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| `a_person` | 0.018 | 0.033 | 0.06 | 0.11 | 0.2 |
| `a_background` | 0.059 | 0.125 | 0.266 | 0.565 | 1.2 |
| mosaic `a` | 0.35 | 0.527 | 0.794 | 1.2 | 1.8 |

**Calibration.** `weak_calib` gives medians over 4 seen16 clips × 3 windows, with 6 clean context latents and 4 history latents. The draw-grid figures are single draws on clips 03, 14, 05 and 07.

- **Background ΔE vs amplitude:** ≈ 8 × (`a_person` + `a_background`) along DARK / DRIFT (7–8× over the range). A random direction gives ≈ 4× less.
- **Colour max (0.2 + 1.2):**
  - background ΔE: DARK 10.1, DRIFT 9.7, random 2.3;
  - person ΔE: 3.7, 3.8 and 0.95 respectively.
- **Colour min (0.018 + 0.059), DARK / DRIFT:** background ΔE 0.60 (p10–p90 0.42–1.0).
- **Person ΔE** with `a_background` = 0, DARK / DRIFT:
  - 0.32 at 0.018;
  - 0.54 at 0.03;
  - 1.44 at 0.08;
  - 3.6 at 0.2;
  - random directions give ≈ 4× less.
- **Mosaic background texture:**
  - a = 0.35: median +0.29 (p10–p90 0.15–0.43), and +0.37 to +0.48 on the draw-grid clips;
  - a = 1.8: +3.7 to +4.3 on the draw-grid clips.

**Draw grids:** [clip 03](../../deliverables/corruption-plan/draws-clip03.png), [clip 14](../../deliverables/corruption-plan/draws-clip14.png), [clip 05](../../deliverables/corruption-plan/draws-clip05.png), [clip 07](../../deliverables/corruption-plan/draws-clip07.png).
- Columns: the clean GT decode, a real rollout frame near drift onset, then the 5 quantile draws from min to max. The colour rows pair `a_person` and `a_background` at the same quantile.
- Rows: DARK, DRIFT, random, mosaic.

**Design.**
- The colour max stays high so the model can be steered.
- The mosaic min is set so its weakest draw is still visible (median background texture +0.3). The colour min sits at a median background ΔE of 0.6.
- The newest history slot is corrupted in ≈ 50% of windows: 0.7 × (0.4 + 0.3 + 0.3 × 0.1) ≈ 0.51. The 0.1 is the share of trailing windows whose n covers every real slot.

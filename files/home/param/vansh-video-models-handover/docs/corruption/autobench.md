# AutoBench

A benchmark for background drift and artefacts in long autoregressive (AR) talking-person videos. It needs no ground
truth: every video is scored against its own first frame, the sink. Two metrics, M1 background colour drift and M2
background texture drift, plus an artefact detector that finds, times and types local background artefacts.

## Why

An AR video model generates a long video a few frames at a time, each conditioned on its own previous output and on a
real first frame, the sink. Small errors feed back, so over minutes the wall changes colour, turns noisy or smooth, or
grows coloured bands and mosaic. Because the reference is the video itself, any clip, seed or checkpoint can be scored
without the real continuation.

## M1 and M2

For each video:

1. **Sample** one frame per second, t = 0, 1, …, 299 s for a 5-minute clip. Frame 0 is the sink.
2. **Segment** the person in every frame (BiRefNet-portrait, threshold 0.5). Erode the person and background masks by
   7 px so the outline belongs to neither.
3. **Background** = pixels that are background in both the sink and frame t, so both frames are compared over the
   same pixels.
4. **M1 colour.** CIE76 ΔE between the background's mean CIE Lab colour in frame t and in the sink.
5. **M2 texture.** Fine detail = the Lab image minus its Gaussian blur (σ 1.5 px); its energy E is the mean squared
   residual over the background. M2 = ln(E_t / E_sink).
6. **Summarise.** Keep the 1 fps curves and their mean over each 60 s block (minute 1 excludes the sink second).

| | Metric | Unit, range | Reads as |
|---|---|---|---|
| M1 | Background colour drift | ΔE, ≥ 0 | 0 = same mean colour as the sink. About 2.3 ΔE is a just-noticeable difference. |
| M2 | Background texture drift | log ratio, signed | 0 = same fine detail as the sink; above 0 noisier, below 0 smoother. ±0.69 = twice / half. |

Both are region means: a global shift shows fully, a small local patch is diluted. The artefact detector covers local
patches.

**Derived scores:** clips over = M1 > 1.0 for ≥ 5 s in a row; seconds over 1.0; minute-5 ΔE = M1 over 240–299 s; end
ΔE = M1 over 293–297 s; mosaic criterion = some minute 2–5 with block-mean M1 > 2 and M2 > 1.

**Real footage** does not score 0 (light flickers, people move), so read a score against real footage of the same
kind, not against 0. Real studio clips (fixed camera, grey wall) sit at M1 0.1–0.2 ΔE and M2 +0.2 to +0.3, flat over
5 minutes; handheld phone footage with changing light is already at M1 1–2 ΔE.

**Reading it** (rules of thumb): M1 well above the real range and rising = background colour drift. M1 and M2 jumping
together = local texture artefacts such as mosaic or speckle. M1 rising without M2 = a region being recoloured. M2
below 0 = the background has been smoothed. Minute means hide onsets: look at the 1 fps curves and the frames.

## Artefact detector

Finds local background artefacts that M1 and M2 dilute (coloured bands, stripes, mosaic blobs, regional casts, grey
texture), each with an onset, an offset and a type. CPU only. Code `/home/vansh/eval/ar_artefacts/` (`extract.py`,
`feat.py`, `detect.py`; method in `REPORT.md`); latest use `/home/vansh/eval/ar_artefacts_20619/` (`job.sh`, `run.py`).

1. **Frames:** the same labelled side-by-side mp4s as M1/M2, one frame per second, 256 × 480 px.
2. **Background:** person = torchvision DeepLabV3-MobileNetV3 person probability > 0.5, dilated 8 px (a fast CPU
   stand-in for BiRefNet); valid pixels are background in both the sink and frame t.
3. **Cells:** 8 × 8 px. A cell is used if ≥ 75% of its pixels are valid; only rows above y = 360 px (floor and hands
   dropped).
4. **Per cell:** global shift g = per-channel median over cells of the cell-mean Lab(t) − Lab(sink); local colour
   change = |cell change − g| in ΔE; texture = ln((E_t + 1) / (E_sink + 1)), E as in M2, +1 as a noise floor.
5. **Flag** a cell if local change > **6 ΔE** or texture > **1.0**; keep 8-connected groups of ≥ **4** cells; A_t = kept
   cells in second t.
6. **Seconds:** strong = A_t ≥ **8** or |g| > **2.0 ΔE**; weak = A_t ≥ 4 or |g| > 1.5.
7. **Episode:** starts at ≥ **5** strong seconds in a row (gaps ≤ 2 s closed), lasts while weak seconds come within
   **10 s** of each other.
8. **Type** (class with the most kept cells per strong second): **streak** = change gradient energy vertical /
   horizontal ≥ 3 (horizontal bands); **mosaic** = isotropic, chroma change ≥ 4, texture ≥ 1; **cast** = chroma ≥ 4 but
   smooth, or only |g| > 2; **other** = chroma < 4 (grey speckle, wall texture).

Thresholds were set so real footage (55 GT clips) gives 0 episodes. Output per episode: onset, offset, type, region,
peak second and area, max |g| (`episodes.csv`), plus frame sheets (onset − 10 s, onset, peak with kept cells outlined).
Limits: onsets run 10–30 s late; artefacts on the person and the floor are not covered; hair or hands leaking past the
person mask can fire.

## Running it

```bash
.venv/bin/python /home/vansh/eval/arbench.py IN_DIR OUT_DIR [--gpus 0,1,...] [--clips 4,5,...] [--band 128]
```

- **Input:** full-quality side-by-side mp4s (real left, generation right, label band on top; only the generated half
  below the band is read), named `NN_*.mp4`, plus a `manifest.csv` with `index` and `uuid`. Never score compressed
  copies: compression wipes the fine detail M2 measures.
- **Output:** one JSON per clip (1 fps curves, minute means, clip means); M1 = `color_bg`, M2 = `tex_bg`. Existing
  JSONs are skipped, so outputs from several nodes can be merged. Real-half variant:
  `/home/vansh/eval/bench/seen16_14922/arbench_gt.py`, same arguments.
- **Compute:** one worker per GPU, about 20 s per clip after loading.
- **Models:** `ZhengPeng7/BiRefNet-portrait` and InsightFace buffalo_l (`~/.insightface/models/buffalo_l`). Extra
  packages in `$ARBENCH_PKGS` (`/tmp/arbench_pkgs`):
  `pip install --target /tmp/arbench_pkgs --no-deps insightface onnxruntime-gpu onnx ml_dtypes opencv-python-headless scikit-image lazy_loader imageio tifffile timm kornia kornia_rs`.
- **Artefact detector:** copy `/home/vansh/eval/ar_artefacts_20619/`, set `ROOT`, `FULL`, `RUNS` in `detect.py`, then
  `extract.py jobs.tsv N` (lines: out_npz, mp4, x offset 256 gen / 0 GT) and `run.py`.

## AutoBench+ (formerly ARBench)

The earlier five-metric benchmark: M1, M2 plus M3 person colour drift (M1 on that frame's eroded person mask), M4
person texture drift (M2 on the person) and M5 identity (ArcFace cosine, InsightFace buffalo_l, of the largest face vs
the sink's; 0.8 = same appearance). A relic, kept only to read past results; not used going forward. Same scorer
(`/home/vansh/eval/arbench.py`, `/home/vansh/eval/bench/ar_report.py`).

- **M3, M4 dropped:** the person region is each frame's own mask, which includes hands and arms, so gestures, hands over
  the face or shirt, head turns and leaning change the region and move the score without drift.
- **M5 dropped:** no occlusion handling: a hand over the face reads as drift, undetected faces are silently skipped,
  and a hand in the sink frame corrupts the reference for the whole clip.

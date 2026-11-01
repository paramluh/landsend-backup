# Uploading samples

Every sample on HF is a **labelled** side-by-side, never a raw `sample_<n>_sxs.mp4`. The scripts are in `upload/`. In `label.py`, `ROWS` can be a list of csv rows, and REFS_ROOT can be a `pick.py` dir.

`production/` keeps one yaml per inference path as a template (e.g. `sink_universal.yaml`). For a rollout, copy it, edit the copy (step, csv, rows, bank, output_dir), run it, then delete the copy. The run dir keeps its own copy of the yaml, and that is the one that gets uploaded.

## Where samples live

- The sampler (`sink_universal.py`, `refs_universal.py`, `cached_universal.py`) writes into the yaml's `output_dir` (with `{step}`): `samples_<n>.mp4` (GT), `sample_<n>.mp4` (gen, with audio), `sample_<n>_sxs.mp4`, `sample_<n>_latents.safetensors` and a copy of the yaml. `n` is the csv row and `seed = yaml seed + n`.
- On m7 these are under `~/samples/<run>/...`, e.g. `~/samples/studio-9-sinks-proper/run2-step11935/{seen,beatles}/`.

## The labelled video (`label.py`)

Here is an example frame (seen row 9 from studio-9-sinks-proper step11935 raw), at m7 `~/eval/samplefix/frames/v2_seen_09.png`. A sinks run's video is 512 x 992, top to bottom: the band (128), GT | gen (480), the sinks strip (384).

- **Top:** a 128 px black band over the 512 px width:
  - Line 1 is `<seen|unseen>  ·  <kind>  ·  <hours> h train` (DejaVuSans-Bold 22).
  - Line 2 is the full uuid (DejaVuSansMono 18).
  - Line 3 is `seed <seed>  ·  <SET> #<csv row>` (DejaVuSans 17).
  - The captions are `GT` and `<GEN label>` over each half, at y = 96 (Bold 21).
- **Middle:** GT | gen, 256 x 480 each.
- **Bottom (refs runs only):**
  - **Sinks runs:** these use a v2 bank (`<bank>/<uuid>/property_<N>/`, e.g. `sink_universal`), and REFS_ROOT is a `pick.py` dir. The bottom is a static 512 x 384 strip of all 9 refs the model got, across the full width.
    - Refs 1–5 are on the top row and 6–9 are centred below.
    - Each tile is 102 x 192, keeping the ref aspect, with its slot number in white on black at the tile's top-left.
    - A missing property is a black tile labelled `N missing`.
    - Each ref is the Wan2.2-VAE decode of the picked latent (`image.py` → `sample_<n>_refs.png`, which `label.py` crops into tiles).
  - **Older `refs_universal` runs:** REFS_ROOT is the refs dir. The bottom is a 480 px panel with a 3 x 3 grid under the gen window, refs from this chunk outlined red, and the text `k/9 from this chunk` on the left half.
- **Encoding:** H.264 CRF 18, preset medium, yuv420p, audio copied from the gen.
- **Field sources:**
  - `hours` is the sum of `duration` over the uuid's rows in the run's **full** training csv, and `seen` means hours > 0. For node-sharded runs, `~/aloe-studio/shard.csv` is only one node's shard. Use the source csv, e.g. `csvs/aloe-studio-no-blackbands-high-quality.csv` for the studio-v2 runs.
  - `kind` comes from the eval csv.
  - `seed` is the yaml `seed` + row.
  - `GEN label` is `step<step> <ema|raw>`.
  - `SET` is the clipset, e.g. `studio-seen` or `studio-beatles`.
- **Outputs:** files are named `NN_<hours>h_<seen|unseen>_<kind>_<actor8>.mp4`, and `manifest.csv` holds the file, the generator and all the row fields.

## HF layout: dataset `vanshjjw/aloe-samples` (private)

- **Single run:** `samples/<run>/step<step>-<ema|raw>-<clipset>/`. It holds the labelled mp4s (faststart), `metadata.csv` and the rollout yaml.
- **Comparisons:** `samples/comparisons/<runA>-vs-<runB>/...`, plus a `README.txt`.
- **`metadata.csv` columns:** `file_name,run,step,weights,clipset,clip,clip_id,actor,kind,category,speaker,duration,seed,train_hours,recording_id`, plus `refs_present` or `refs_in_chunk` for refs runs.
  - `clip` is the csv row, `clip_id` is the full uuid and `actor` is its first 8 characters.
  - `duration` is the csv chunk duration.

## Commands (studio-9-sinks-proper step11935 raw, seen set)

Run these from `upload/`. The yaml's relative paths resolve against `production/`.

```bash
# 1. refs runs with a v2 bank only: the refs each rollout got (needs dataset + bank, so run where the rollout ran; CPU)
python pick.py RUN_DIR/sink_universal.yaml PICKS/seen
#    then decode them to RUN_DIR-style grids PICKS/seen/sample_<n>_refs.png (one GPU)
python image.py PICKS/seen
# 2. label (ROWS = the yaml's rows, or start,end)
python label.py ../csvs/aloe-studio-seen-eval.csv ../csvs/aloe-studio-no-blackbands-high-quality.csv \
    9,75,42,14,6,1,2,10,21,40,4,41,66,28,3,27 RUN_DIR OUT/seen "step11935 raw" studio-seen 2026 PICKS/seen
# 3. look at a frame
ffmpeg -ss 5 -i OUT/seen/09_*.mp4 -frames:v 1 /tmp/f.png
# 4. stage (faststart remux, metadata.csv, yaml), replace the HF folder in one commit, verify size + sha256
HF_TOKEN=$(grep '^VANSH_HF_TOKEN=' ../.env | cut -d= -f2-) \
    python upload_samples.py OUT/seen studio-9-sinks-proper 11935 raw studio-seen RUN_DIR/sink_universal.yaml
```

Here `RUN_DIR` is `~/samples/studio-9-sinks-proper/run2-step11935/seen`.

`upload_samples.py` deletes every other file under the prefix. It prints `verified N/N ... remote == local: True`, and the upload is done only when that line says so.

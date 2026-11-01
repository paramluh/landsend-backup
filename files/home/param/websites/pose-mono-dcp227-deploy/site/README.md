# shared/genweb: the pose playground site

`monoplex-pose/` and `duplex-pose/` both use this one site. The page reads `?model=monoplex` (the default) or `?model=duplex`. The duplex view also has the Live-mic call (`livedup`). How-tos for each mode are in [`../../monoplex-pose/README.md`](../../monoplex-pose/README.md) and [`../../duplex-pose/README.md`](../../duplex-pose/README.md).

Keep every file in this folder together: the server and workers use their own directory as the queue root.

## How it fits together

```
browser ── tunnel ── front.py :9205 (always up) ── server.py :9206|:9207 (no GPU, zero-downtime swaps)
                                                     │  files only, no HTTP to workers
                     runs/<id>/request.json + READY ─┤
 worker.py (monoplex, 2 GPUs) / worker_dup.py (duplex) claim READY -> CLAIMED.<w>, write status.json + b.wav, words.json, tokens.npz, feed.jsonl
                     render/jobs/<id>.feed.jsonl ────┘──> render service (render_flex.sh / render.sbatch) -> render/out/<id>.mp4
```

## Files

| File | Role |
|---|---|
| `index.html` | The whole page: form built from the worker's schema, queue, History, batches, Live mic. |
| `server.py` | aiohttp app: `/api/schema`, `/api/gen`, runs, uploads, and the static run outputs. It mounts the APIs below. |
| `front.py`, `app_sup.sh`, `app.sbatch` | An always-up front on :9205 that proxies to whichever app port is in `APP_PORT`. To deploy, `touch DEPLOY`: the new app starts on the other port, the front switches once it is healthy, and the old app drains. |
| `tunnel.sbatch` | Cloudflare quick tunnel to :9205, which writes `URL.txt`. Run it on the same node as `app.sbatch`. |
| `prefs_api.py` | `/api/prefs`: the user's last-used settings, kept on the server (`state/prefs.json`) so they survive tunnel/origin changes. |
| `history_api.py`, `genids_api.py`, `dupui_api.py`, `voiceclip_api.py` + `voiceclip.py` | History tab and thumbnails; `G<n>`/`B<n>` run ids; duplex batch zip; voice-clip preview. |
| `livedup_api.py` + `livedup_import.py` | Live duplex calls: the page streams the mic to the live pose-duplex server, then the call is imported as a run (History/render). |
| `worker.py` / `worker_dup.py` | GPU workers: monoplex engine (`aloe-pose-monoplex-infer`) and duplex engine (`aloe-pose-duplex`). Each writes `SCHEMA.json` / `DUPLEX_SCHEMA.json` + `OPTIONS.json`. |
| `SCHEMA.json`, `DUPLEX_SCHEMA.json`, `OPTIONS.json` | Example worker-written schemas, so the page renders before any worker is up. |
| `weights/*.json`, `MONO_CK.json` | Example extra-model registry entries (see monoplex README). |
| `render*.sbatch`, `render_flex.sh`, `flex_lib.sh`, `render_vansh_unit.sh` | The render queue: avatar video renders through the video team's `render_service` (`$CANOPY_ROOT/video/dmd-rt`, not in this repo). |
| `unit.sbatch`, `hot_unit.sh`, `mono_unit.sh`, `worker.sbatch`, `dup_hold.sbatch`, `dupui.sbatch` | GPU-pair launchers: one hot worker per pair, with an optional render lane. |
| `pool.sh`/`pool.sbatch`, `elastic_ctl.py` | Optional: backfill spare GPUs with worker/render pairs; elastic worker count per model. |
| `pose_scorer.py`/`scorer.sbatch`, `flash_worker.py`/`flash.sbatch`, `stick.py` | Side jobs: pose stats per run, flash + beep burn-in, stick-figure render. |

Runtime dirs that are not committed: `runs/ uploads/ render/ state/ logs/ weights_live/ voiceclips/ openings*/ people_extra/` and the `URL.txt`/`APP_PORT` files.

## Run the site alone (no GPU)

```bash
srun -c 2 --mem=4G --pty bash                      # a compute node, never the login node
pip install aiohttp numpy                          # or use your venv
cd shared/genweb && python server.py --port 9205   # http://127.0.0.1:9205 ; ?model=duplex for the duplex view
```
With no worker running, runs stay queued. Start a worker (see the mode READMEs) to process them.

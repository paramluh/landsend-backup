# DUPUI audit: duplex mode (?model=duplex) vs monoplex, 10-26 ~05:25 UTC

**Method.** I read index.html, server.py, history_api.py, worker.py and worker_dup.py, and the 6 duplex runs on disk (all from 10-25, before the batch, stream, video-weights and GT features existed).

**Headless check.** `genweb/dupui_check.py <URL> 20261025-170246-535f` ran on slinky-18 via `srun --jobid=30452 --overlap`, with `LD_LIBRARY_PATH=/data/home/amartyavarma/chromelibs/root/usr/lib/x86_64-linux-gnu` and the pagecheck venv. It is read-only and never presses Generate. Screenshots are in `logs/dupui-audit-<w>.png`.

What the check confirmed at 1280, 1440, 1920 and 390 px:
- Duplex mode loads with 0 page errors and no horizontal scroll.
- All 7 settings tabs and the Generation tab (Seed, Samples, Stream live, Show GT, Video weights) are present.
- The badge reads v2026.10.25-18.
- The URL carries all 56 duplex knobs (none missing).
- The run view shows the A and B word lanes; the batch grid and zip link work; History shows duplex rows.

## Feature by feature

| feature | duplex today | where |
|---|---|---|
| Generation tab (settings + batch tiles, click/arrows) | **works**: model-agnostic DOM move + `batchTag`/strip | index.html batch block (751-989) |
| Batch grid (live fill, PT stamp, zip) | **works**, but: **card text mixes A+B words** (server joins every non-prefill word); zip has only the B-voice mp4s, **no A / mix audio** | server.py:366 `card()`, :452 `batch_zip()` |
| History (search, filters, re-run) | **works**: `pTxt` / `hay` know duplex (conv, side, person). **No Model filter** (text search "duplex" works). Re-run goes through `runVals` (handles conv/side/a_after/person) | index.html history block; history_api.py |
| Stream live (default OFF) | **works**: server writes `stream:false, no_live:true` unless the user switched it on; worker_dup honours it (`req.get("stream") is True and not req.get("no_live")`) | server.py:255-260, worker_dup.py:333 |
| Video weights Param / Vansh / Both | **plumbing works, untested on a duplex run**. Vansh → worker_dup queues `render/jobs_vansh` (worker_dup.py:411) and sends `weights:vansh` on live start. Both → server forces no_live and writes `compare.json`; `comparer()` copies feed.jsonl → `jobs_vansh/<rid>.vansh.feed.jsonl` → `render/out/<rid>.vansh.mp4`, model-agnostic | server.py:262-272, 316-343 |
| Show GT (default OFF) | **works for all 20 avatars** because `gt-<person>.mp4` already exists for all 20 (render/out + out_seq); `run_summary` takes `person` from `req.person` for duplex. **worker_dup ignores `req.gt`**, so a missing GT is never queued (monoplex: worker.py:260 `gt_render(only={person})`). Semantics: GT = the avatar actor's own 12 s clip, **not** the conversation's real side-K speaker | worker_dup.py `run()`; server.py `run_summary` |
| Version badge / changelog | **works**; `running.speech` already says "duplex: export-498". After the duo deploy, set it to say "both sides generated" | CHANGELOG.json |
| Reproducible links | **works** (every schema row in the URL), but a **Model switch drops the page flags** (gt / stream / vw): `$('mdl').onchange` replaces the query with just `?model=duplex`, and `urlLoad` then resets GT_ON / STREAM_X / VW_X. The check confirmed `before ['gt=1','stream=1','vw=both'] → afterMono [None]`. Affects monoplex too | index.html:432 |
| Run view audio | **missing**: only `b.wav` ("B's voice") is shown; `a.wav` / `mix.wav` exist but there is no player. The avatar video carries B's voice only, so **A is never heard** in the main player, batch cards or modal | index.html:504 `resetDetail`, :549 |
| Two-sided words (ch A/B) | **works**: `renderWords` splits by `ch` into A / B lanes (confirmed) | index.html `renderWords` |
| Avatar on seq engine | **works**: the duplex feed → `render/jobs` → `render2.sbatch` (videort `rt/render_multi.sh`, RS_ENGINE=seq); the flasher burns the flash at `prefill.s`. Duplex run 535f was promoted to its out_seq re-render and has `.flash.mp4` | worker_dup.py:411-415, server.py flasher |
| /api/gen duplex request | passes `batch`, `stream`/`no_live`, `video_weights`, `gt`; whitelists `conv, side, a_after, person` only. **A new top-level field (e.g. `dup_mode`) would be dropped**. `params` pass through verbatim | server.py:251 |
| Labels | "Duplex (hears A)" (option text reset every 4 s by `loadSchema`, gomodel, mdl tooltip) is wrong once duo is the default | index.html:269, :346, :420 |
| Throughput | ONE duplex worker (job 30533, slinky-18 GPUs 4,5, one engine), so a duplex batch runs **one sample after another**. The gomodel tooltip says the samples "run in parallel". Duo = 2 states per run, so each run is slower | job 30533 |

## (a) Changes inside worker_dup.py (dupui-owned)

1. **`dup_mode` as a schema row carried in `params`.** This needs no server or core-page change. `engineParams()` sends every non-`DUP_SKIP` row in `params`, and the server passes `params` through verbatim. `runVals` merges `params`, so the knob gets exact links, Load settings and History re-run for free. The worker pops the knob before the engine-knob check (otherwise you get `ValueError: unknown engine knobs`):
   ```python
   # schema_rows(), after the a_after row:
   {"name": "dup_mode", "label": "A side", "group": "Conversation", "type": "enum", "default": "duo", "choices": ["duo", "replay"],
    "choice_labels": {"duo": "both sides generated (two model states, each hears the other)", "replay": "B generated, A = the real recording (replay)"},
    "hint": "duo: two states of the model on one weight copy, each fed the other's generated audio (poseduplex/samples gen_pd.py); replay: A = the conversation's real other side (or silence after the opening, see 'A after prefill'); Call 1086 is replay only"},
   # run(): before `bad = [...]`
   pr = dict(req.get("params") or {})
   mode = str(pr.pop("dup_mode", None) or req.get("dup_mode") or "duo")
   if cid == "call1086": mode = "replay"
   ```
   Also put `dup_mode` and the line delay in `summary.json` and the headline.
2. **GT on demand.** All 20 GT mp4s exist, so this is only a safety net. Port worker.py's `gt_render(only={person})` (worker.py:211-229) and call it when `req.get("gt") is True`. Decode with the engine's Mimi the way gen_pd does: `eng.mimi.decode(torch.as_tensor(codes, device=d0)[None]).audio_values[0, 0]`. Check that this decode equals worker.py's `self.mimi.decode` before using it (same maths).
3. **Duo outputs, same contract.**
   - `b.wav` = state K's audio.
   - `a.wav` = channel A as fed, i.e. the partner state's generated decode delayed by the line delay.
   - `mix.wav` = `a_fed + b`.
   - `words.json`: the partner lane's words are `ch:"A"`, with `t0` shifted by the delay so they match what B heard.
   - `feed.jsonl` = B only, so the avatar stays B.
4. **Optional.** A real conversation GT for c0-c7: a render job `gt-<cid>-s<K>-<person>` from `gtpose_K`, `depB_K - OFF` and `stem_K` (all in convs/cK.npz). This would put the real speaker's GT next to the generated one, unlike the actor clip. It is a new render only and touches no maths.

## (b) A new appended block `<!-- BEGIN dupui -->` (plus an optional `dupui_api.py`)

All of these hook globals that the core resolves at call time: function declarations, `const` arrays and objects that can be mutated, DOM `onchange`, and `window.vw*` hooks.
```js
// conversation tab + short labels (const arrays/objects are mutable)
TABS_DUP[1][2].push('dup_mode'); PLAIN.dup_mode='A side'; SHORT.dup_mode={duo:'both sides generated',replay:'real A (replay)'};
if(SCHEMA&&MODEL==='duplex') build();
// old duplex runs (no params.dup_mode) were replay: keep their exact links / re-run truthful
const _rv=runVals; runVals=function(r){ const o=_rv(r), rq=r.request||{}; if(o.m==='duplex'&&!('dup_mode' in (rq.params||{}))) o.v.dup_mode='replay'; return o; };
// labels
const _um=updModelUi; updModelUi=function(){ _um(); if(MODEL==='duplex') setTxt($('gomodel'), VALS.dup_mode==='replay'?'Duplex (B generated, real A)':'Duplex (both sides generated)'); };
// a_after only means something in replay: disable its row in duo
// run view: wrap showRun -> for duplex runs add a "Conversation" section (mix.wav + a.wav players) and a "Play A with the video" toggle
//   that keeps an <audio src=a.wav> synced to #dav video (own sync: vwSync mutes its B side, so it cannot be reused for audio)
const _sr=showRun; showRun=async function(id){ await _sr(id); /* add players once per id when files have a.wav / mix.wav */ };
```
- **Model switch keeps the flags.** Override `$('mdl').onchange` from the block so it builds the query with `gt` / `stream` / `vw`. This changes monoplex too, so tell aloedemo.
- **Batch.** Wrap `window.vwCard` to add an A-audio button on duplex cards. A `dupui_api.py` route `/api/dupzip/{bid}` (mp4 + mix/a/b wav), mounted with one `setup(app, globals())` line in server.py, can provide a "Download all (with conversation audio)" link through a `window.vwBatch` wrapper.

## (c) Changes in aloedemo's existing code (send as patches)

1. **Batch card text: B words only, plus A separately** (server.py:366):
   ```python
   c["text"] = " ".join(str(w.get("w", "")) for w in ws if isinstance(w, dict) and not w.get("prefill") and w.get("ch", "B") == "B")[:1200]
   c["text_a"] = " ".join(str(w.get("w", "")) for w in ws if isinstance(w, dict) and not w.get("prefill") and w.get("ch") == "A")[:1200]
   ```
2. **Zip carries the conversation audio for duplex** (server.py `batch_zip`, after the mp4 writes):
   ```python
   for n in ("mix.wav", "a.wav", "b.wav"):
       if (c.get("request") or {}).get("model") == "duplex" and os.path.exists(f"{RUNS}/{c['run_id']}/{n}"):
           z.write(f"{RUNS}/{c['run_id']}/{n}", arcname=f"{bid}_{i + 1:02d}_seed{c['seed']}_{c['run_id']}_{n}")
   ```
3. **Option text and tooltip** (index.html:420 and :269): `'Duplex (hears A)'` → `'Duplex (both sides)'`. `loadSchema` resets it every 4 s, and `setInterval(loadSchema, 4000)` holds the original function, so a wrapper cannot fix it.
4. **Only if `dup_mode` goes top-level instead of in `params`:** add it to the whitelist at server.py:251 and to the `go()` body at index.html:427, and add `put('dup_mode', rq.dup_mode)` to `runVals` at :496. Not needed with plan (a)1.
5. **Optional.**
   - A History Model filter (`F` / `pass()` are private to the history block).
   - `"dup_mode"` in history_api.py `KEYS`.
   - The gomodel tooltip (:262): "run in parallel" is false for duplex while it has one worker.

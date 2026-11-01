# MoT 14B, audio-in MoT, ASR adapters, PPM-Voxtral: code registry

This is a snapshot of code, configs and docs only, copied read-only from the live working dirs on 10-27 at about 20:00 PT.
It holds no checkpoints, data, caches or logs, apart from small result docs written by the runs' own tests.
Each sub-dir has its own README with what it is, its job ids, init, data, schedule, results, and where the checkpoints live.

| dir | source | what |
|---|---|---|
| `mot/` | `ppm/mot` | MoT 14B pretraining (3 Qwen3-14B experts per layer, INTL layout). Jobs 28071 to 30289, paused at step 14839. Includes the guard kit, the watchdog and the live engine. |
| `motvox/` | `ppm/motvox` | The MoT 14B continued with audio-in podcast rows: the user is heard through Voxtral and the motasr2 adapter. Job 31626, **LIVE at the time of the copy**. |
| `mot-asr/` | `ppm/mot-asr` | The Voxtral ASR adapter trained on the frozen MoT 14B: motasr (j30741, base w-10830) and motasr2 (j31090, base w-14810; best w-1920). |
| `asr-adapter-14b/` | `ppm/asr-adapter-14b` | The Voxtral ASR adapter trained on the frozen INTL 14B (next-arch w-6242). asr14b j23526/j24039, final w-480. |
| `asr-adapter/` | `ppm/asr-adapter` | The Voxtral ASR adapter trained on the frozen PPM 32B. Runs: asrppm / asrppm-nonorm (dcp-11488), asrppm-w15973, asrppm-w16861 ("ASR16861"), asrppm-811-17850. |
| `ppmvox/` | `ppm/ppmvox` | PPM-Voxtral, a full fine-tune of PPM 32B with Voxtral user audio. Runs v1, v3 (w-17127, the pre-Aloe base), v5 and v5b. |
| `voxtral-libs/` | `qwen-1ch/stage4-asr2`, `qwen-1ch/infer/s4{fast,self}`, `models/voxtral-realtime` | The Voxtral front-end, the adapter class, the audio decode and the MLS stream that every ASR trainer imports, plus the Voxtral model config (no weights). |

## Lineage, short form
```
qwen-1ch stage4-asr2 j19174 adapter w-3695 (sha 45162ffc)  -- the adapter init for every ASR run below
  +-- PPM 32B dcp-11488  -> asrppm-nonorm (j21990, final s1972, best norm CE 0.2549)
  +-- PPM 32B w-15973    -> asrppm-w15973 (j26064, w-1441, best 0.2629)
  +-- PPM 32B dcp-16861  -> asrppm-w16861 (j28311 -> j28462, w-1346, best 0.2662) -> PPMVOX v3 init (j29005) -> w-17127 -> Aloe FTs
  +-- PPM 8,1,1 17850    -> asrppm-811-17850 (j29676, w-835, best 0.2878)      -> PPMVOX v5 (j29863) -> v5b (j30104)
  +-- INTL 14B w-6242    -> asr14b (j23526 -> j24039, w-480, 0.3158)
  +-- MoT 14B w-10830    -> motasr (j30741, paused @1220, best 0.2657)
  +-- MoT 14B w-14810    -> motasr2 (j31090, stopped @2094, best w-1920 0.2527)
                               +-- MoT dcp-14839 + motasr2 w-1920 -> motvox (j31626, running)
```
"norm CE" is the transcript normalised word CE on 384 MLS dev clips (lower is better).

## Copy rules applied
- Included: `*.py *.sh *.sbatch *.env *.md *.patch *.js`, small `*.txt` (100 KB or less), and an allow-list of small JSON records: run `cfg.json` / `status.json`, `stop_rule.json`, `HF_BACKUP*.json`, base GATE/VERIFY records, presets and plan summaries. Job-id marker files (`JOB`, `TRAIN_JOB`, `LAUNCHED`, `CURRENT_JOB`) are included too.
- Excluded: checkpoints, exports, adapter weights, data and index files (jsonl/tsv/npz/npy/arrow/flac/wav), logs, `__pycache__`, `*.bak*`, `*.orig`, lock and state dirs, stream shards, and every file over 2 MB.
- Excluded for safety: the `control/` dirs. motvox's `control/` is live; the copy never went into it. For the ASR runs, only `stop_rule.json` was taken.
- `asr-adapter/` and `asr-adapter-14b/` had their own README in the source. It is kept as `README.source.md`, and the registry README sits next to it.
- Secrets: see `REDACTIONS.md` (the scan found nothing).
- Paths inside the files are the cluster's absolute paths (`/data/home/amartyavarma/...`, `/scratch/amartyavarma/...`); they were not rewritten.

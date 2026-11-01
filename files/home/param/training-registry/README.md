# Training registry (private)

Snapshot as of 2026-10-27 20:35 PT. It holds the training code, a data catalogue and the full model lineage of our speech/pose model work.
- It is a copy for reference. Every source was copied read-only, and the code still names the original absolute paths, so it does not run from here as is.
- Checkpoints and datasets are not included. They are linked: HF repo links in `lineage/` and `data/`.

## What's where

| Part | Contents |
|---|---|
| [lineage/](lineage/) | **Start here.** `LINEAGE.md`: Mermaid graphs from the base models (Qwen3-32B, Qwen3-14B, Mistral Medium 3.5 128B, Voxtral) through PPM 32B, the 8,1,1 run, the 14B family, MoT and every fine-tune, plus a "deployed where" table. `RUNS.md` / `runs.json`: 161 training runs with job ids, init, data, steps, final evals and outputs (HF links), each cited to a file:line. |
| [data/](data/) | `DATA_CATALOGUE.md` / `datasets.json`: 985 entries. 104 are researched in depth: contents, encoding (Mimi lufs23-local vs old, SAM vs raw, pose tokenizer), size, location link, provenance, held-out rule, which runs used it, quirks and a cited **opinion**. The rest are listed from the HF org inventory. |
| [code/core/](code/core/) | Pretraining trainers: PPM 32B main, the 8,1,1 continuation (incl. the fresh-pose ingest and today's s2p + G0COND pose run), the 8,1,1 decay branch, INTL 14B (next-arch), ppm14b, the Megatron port, MoT 14B, PPM-Voxtral, Mistral-128B PPM, interleave variants. |
| [code/mot-asr/](code/mot-asr/) | MoT 14B, the audio-in MoT (motvox, training now), the Voxtral ASR adapters (PPM 32B asrppm*, asr14b, motasr / motasr2), PPM-Voxtral and the Voxtral front-end libraries. |
| [code/finetunes/](code/finetunes/) | 49 fine-tune recipes in 6 groups: monoplex (Aloe handover mono rows), Xabi v2pure (kept separate), Aloe speech with Mimi on A, Aloe speech with Voxtral on A, si20k, and others (samsep, poseduplex, Rosie, quick-QA, ttconvo). Shared code is stored once; `_tools/materialize.py` rebuilds every recipe byte for byte. |
| [code/early/](code/early/) | Pre-PPM stages: Qwen3-32B single-channel stages 1-4, quick-QA, Mistral 128B single-channel, the E4-E28 / klpre lineage, the imend / tcache variants, the Demeter/Hestia reference and GenghisBench (sim_20q) eval harnesses. |
| Pose work (separate repo) | Private HF `InternalCan/pose-repo` holds the pose trainer, inference engine, playground, eval and render code, plus the pose tokenizer code. |

Each part has its own README and `REDACTIONS.md`. Secret scans of every part found no tokens or keys.

## Open issues these docs surface (see data/DATA_CATALOGUE.md and lineage/RUNS.md)
- **Licences:** licence provenance is unrecorded for most training sources (TikTok pose, the fresh pose deliveries, podcast audio, the handover stores, the voice bank).
- **Mimi encodings are mixed** across sources: lufs23-local-v1 vs the old encode vs mimi-stream-v5.
- **8,1,1 held-out pose eval:** its rows are old-encode TikTok, so the pose-row speech cb0 jumped at 16861, and there is no fresh-data pose eval.
- **Aloe pose FT stores** (monoplex 24272 / mono17850) include single-bleed conversations beyond the 'none' bucket.
- **Delivery 4 pose data** (~333 GB) was never ingested.
- **No HF backup:** some older FT checkpoints are GCS-only or have none.

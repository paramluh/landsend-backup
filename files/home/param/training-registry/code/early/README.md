# code/early: speech-model trainers and eval harnesses that preceded PPM

Code, configs and docs (no checkpoints, data, caches, logs, `__pycache__`, `*.bak*`, or files > 2 MB) copied read-only on 2026-10-28
from `/data/home/amartyavarma`. Each family dir has a `README.md` (what it is, runs/jobs, init, data, key results, checkpoints, gaps) and a
`src/` tree that keeps the source's relative layout. Hard-coded absolute paths in the code still point at the original locations.

## Lineage at a glance

```
Qwen3-32B two-lane (qwen-e28-klpre):  E4 -> E25b (mix-RoPE) -> E27 -> E28 j11411 dcp-4550
     -> hera3s2 stage 2/3 on E28 -> klpre (KL speech pretrain) j21059 dcp-532 -> imend text-KL format (qwen-s2-imend, + qwen-s2-tcache)
Qwen3-32B single-channel redo (qwen-1ch), mix-RoPE:
     stage 1 text KL  j4365 -> j4866 w-4000
  -> stage 2 speech + text KL  dev chain -> j7226 dcp-2557 (baseline); final run j7652 -> j7671 -> j7755 -> j7862 w-4289
  -> stage 3 duplex  exp1 j8197 ... j8424 (lr 0.3 from 750) ... j8815 w-5900
  -> stage 4 Voxtral on channel A  j9267/j9411 validation -> hovfix j10242 w-731 -> quick-QA FTs (quick-qa)
Mistral Medium 3.5 128B (mistral-1ch), stock RoPE:
     pre-1ch stage 1 j91..j519 -> 1ch stage 1 j709 w-531 (GenghisBench 20/20) -> stage 2 j7466 -> j7489 dcp-427 (baseline)
Reference: Demeter / Hestia (demeter-hestia-ref)
```

## Families

| dir | what | source |
|---|---|---|
| `qwen-1ch/` | Qwen3-32B 1ch stages 1-2 + GenghisBench / eval harnesses (`README.md`), stage 3 (`README_stage3.md`), stage 4 (`README_stage4.md`) | `qwen-1ch` |
| `quick-qa/` | quick-QA fine-tunes (runs A-H, scaling) on the stage-4 trainer + quick-qa-100k TTS data generator | `s4-qqa`, `qqa-gen` |
| `mistral-1ch/` | Mistral Medium 3.5 128B distillation, pre-1ch and 1ch stage 1, stage 2, inference, probes | `mistral-distill-bf16` |
| `qwen-e28-klpre/` | the 09-30 handoff package of the two-lane E28 / klpre lineage | `ref/qwen-duplex-handoff` |
| `qwen-s2-imend/` | klpre trainer tree with the `<|im_end|>` text-KL format (superset of `ref/qwen-tts-kl-stage2`) | `qwen-s2-imend` |
| `qwen-s2-tcache/` | teacher hidden-state cache for the klpre/imend text micro | `qwen-s2-tcache` |
| `qwen-s2-mem/` | H100 memory/env overrides for klpre-bpad2 | `qwen-s2-mem` |
| `qwen-s2-cfgtest/` | small tests for the 1ch stage-2 trainer config / LR rewarm | `s2-cfgtest` |
| `demeter-hestia-ref/` | Demeter trainer snapshot + our Hestia comparison tooling | `ref/demeter-5q9kusrz`, `demeter-causalsam`, `setup-logs/s3hestia` |
| `setup-logs-trainers/` | trainer / eval / ops scripts from `setup-logs` (no page builders) | `setup-logs` (subset) |
| `_checkpoints/` | HF checkpoint cards (authoritative eval tables), backup tooling, filtered HF repo list | `hf-backup-staging`, `setup-logs/hfrepos/repos.json` |

Secrets: see `REDACTIONS.md` (nothing needed redacting; credential file paths are referenced, never contents).

## Not included (covered elsewhere or out of scope)

- PPM / MoT / pose / Aloe / avatar / demo code (`ppm*`, `aloe-*`, `sam-aloe-ft*`, `sam-pose-server`, `hera-aloe-live`, ...).
- Other teams' reference trees in `ref/` other than Demeter (hera, sam-*, conversation-duplex-atlas, ...).
- Vendored third-party code removed from the copies: `qwen-1ch/src/stage4-decomp/{mc,mc_pkg}/**/mistral_common`, `qwen-1ch/src/hera-ft/fa3-905da40`,
  `mistral-1ch/src/quantisation/stage1_kl0_fp8/vendor/pyfp8` (torchao), `qwen-s2-mem/pydeps`, `qqa-gen/tts_audit/pylib`, node/ffmpeg bundles.
- Result dumps (GenghisBench game JSONs, xbench result dirs, sample outputs) and run dirs (`runs/<tag>/metrics.jsonl`); READMEs quote
  the numbers and give their original paths.

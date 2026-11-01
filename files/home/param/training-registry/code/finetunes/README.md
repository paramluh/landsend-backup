# Fine-tune recipes (training registry, built 10-27 PT / 10-28 UTC)

This directory holds every fine-tune recipe from `/data/home/amartyavarma/ppm/`: code, configs (env / sbatch / run scripts) and docs
(README / STATUS / CFG_DIFF / PROMPT), with one README per recipe. Checkpoints, data, stores, caches, logs, `__pycache__`, `.bak` files
and files over 2 MB were not copied. Sources were read only.

## Layout
- `<group>/README.md`: dataset, lineage, results table and opinion for the group.
- `<group>/<recipe>/README.md`: per recipe:
  - user ask, parent recipe and diff, base checkpoint, data, prompt rule, schedule / LRs;
  - final held-out evals (normalised word CE; pose CE where relevant);
  - exports + HF repos, and the demo / playground it serves;
  - status and gaps.
- `<group>/<recipe>/src/`: the copied files at their source-relative paths.
  - A file that is byte-identical to the group's canonical copy is NOT stored again.
  - A near-duplicate (diff <= 50 % of lines) is stored as `<file>.diff` (unified diff, canonical -> this recipe).
- `<group>/<recipe>/CODE_MANIFEST.tsv`: one row per copied file, with columns:
  relpath, mode (`full` | `identical` | `diff`), stored_at, sha256, bytes, source path.
- `<group>/_canonical/`: the canonical version of every `.py` >= 4 KB shared by 2+ recipes of the group. It is the canonical recipe's copy
  where that recipe has the file, otherwise the most common version.
- `_tools/materialize.py <group>/<recipe> <out_dir>`: rebuilds a recipe's full tree and checks every sha256. All 49 recipes were rebuilt
  and matched byte for byte.
- `_tools/`: also holds the scripts that made this copy: `stage_copy.py`, `prune_data.py`, `dedupe_install.py`, `groups.json` and
  `extract_evals.py`.

## Groups (49 recipes)
**monoplex_aloe_mono**: Aloe handover MONOPLEX pose rows (aloe-pose-mono-8020 family). Canonical copy: aloe-pose-mono-8020.
- aloe-pose-ft (22091, built the pose stores), aloe-uncond-ft (22317), aloe-nopose-ft (22291), aloe-pose-mono-8020 (export-357),
  aloe-pose-ft-sam (SAM-code store build), aloe-pose-mono-8020-sam (staged, never trained), pose6-mono17850, pose6-mono17850sam.
- Best: **pose6-mono17850**, word 1.8976, pose cb0 1.0700, light 2.9462.

**xabi_v2pure**: Xabi's single-speaker SAM pose data (Aloe pose release v2, "v2pure"). This is a DIFFERENT dataset from monoplex.
Canonical copy: aloe-v2pure-main-17850.
- aloe-v2pure-ft (23051), -clip80, the scaling series (-7932, -13533, -16610, -17352, -17641, -main-17850), monoft (orchestration),
  samdata-ft (init conversion).
- Best pose: main-17850, cb0 1.5788 / light 3.6384 (playground mono slot B). Best word CE: 16610, 2.0749.

**speech_aloe_mimi**: Aloe speech duplex with words + Mimi on A (pre-Voxtral), plus the init comparisons. Canonical copy: aloe-v2init-12529.
- aloe-ft (21980), aloe-v2-ft (22383), aloe-v3-ft (never trained), aloe-v2init-8869 / -10351 / -12529, aloe-next-6242.
- Best: v2init-12529, 1.8230. At equal pretraining loss, PPM w-8869 (1.8442) beats the 14B w-6242 (1.8494).

**speech_aloe_voxtral**: Aloe speech duplex with Voxtral latents on A. Canonical copy: aloe-v2vox-raw.
- aloe-m1vox-ft, aloe-vox-ft (stopped @72), aloe-voxall-ft (stopped @11), aloe-v2vox-ft, aloe-v2vox-raw (23081), -raw-w14503,
  -asr-w15973 (26232), -asr-w15973-textB, aloe-nextvox-6242 (14B).
- Best: v2vox-asr-w15973, 1.8075.

**si20k**: spoken-instruct 20k TTS FTs. Modes vox / txt / tin / tdx; bases PPM 12829 / 11488 or 14B w-6242; 1:1 podcast co-train.
Canonical copy: si20k-vox-ppm.
- 9 runs. Best: si20k-tin-ppm, 1.4148 (text in); si20k-vox-ppm, 1.4567 (speech in).

**other_duplex_cotrain_qa**: Canonical copy: aloe-ft-32-samsep.
- aloe-ft-32-samsep: SAM-sep speech FTs, best **vox17127 1.8023**, plus persona / speaker LoRA spin-offs.
- aloe-ft-32-samsep-poseduplex: **1.7876, pose cb0 1.0695 / light 2.9394**, serves the pose-duplex backend.
- aloe-zoom-cotrain, aloe-pose-rosie, qqa-ft, ttconvo-ft.

Held-out sets differ between groups and lineages (480 / 324 / 316-micro Aloe evals, si20k 192 rows, pose sets). Compare numbers only
within a lineage. The speech metric reported everywhere is normalised word CE (`eval_speech/word_ce_norm` in `runs/<run>/metrics.jsonl`).

## Not included (deliberately)
- Demo / serving dirs: `aloe-*-live*`, `si20k-*-live`, `aloe-textb-tts`, `aloe-pose-monoplex-infer*`, `aloe-posedup-live`.
- The inference release bundle `aloe-ft-32-samsep/release/`.
- Pretraining / main runs (pose5, pose6 top level, code-811, MoT, megaport, ...), which other registry parts cover.
- ASR adapter runs (`asr-adapter*`, `mot-asr`) and the checkpoint-conversion dirs (`ckpt-infer`, `ckpt-restore`, `ckpt-weka`).
- Note: `aloe-v3-ft` sits in speech_aloe_mimi by name, but it was a Voxtral-on-A design (built, never trained).

## Secrets
Both scans found nothing, so nothing was redacted; see `REDACTIONS.md`.

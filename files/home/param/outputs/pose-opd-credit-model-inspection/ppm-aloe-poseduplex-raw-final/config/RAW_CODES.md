# poseduplex-raw: RAW Mimi codes (data path), 10-30

The user approved this run on 10-29 at about 23:01 PT. It is job 30074's recipe verbatim, with POSEDUP_ITEMS=pose_ok and 426 steps (plan d3e5e9624ef3e682). The only change is the Mimi code values. Every row's codes_user and codes_assistant become the raw s2h-m1 codes of the mapped row. These are the raw-lane mimi-stream-v5 codes that job 26232 trained on, and they replace the SAM-separated codes.

## What changed
Only `code/data_aloe_posedup.py` changed. `data_aloe_v2vox.py` and `data_aloe_v2.py` are byte-identical to before (trainer sha asserts: 0b06343f… and 59a8f759…). New knobs (defaults = old behaviour, no key, nothing read):

| env | meaning |
|---|---|
| `POSEDUP_RAW_CODES=1` | Needs ALOE_SAMSEP_STORES. `Stores.open` wraps each sam-sep store's `Rows` in `RawRows`, and the sibling index's rows too, so B's voice-clip prefix (cut from a later-chunk row) also gets raw codes. Each row keeps all its sam-sep fields except the code arrays, which come from the mapped raw row (`ALOE_SAMSEP_ROW_MAP` new→old). The raw arrays are mmapped in place from `POSEDUP_RAW_STORES/m1n<k>`. The stores and A2.DIGESTS stay the sam-sep ones, so POSE_INDEX, BANDMASK and framing are unchanged. |
| `POSEDUP_RAW_STORES` | default `/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores`, read in place (same as ALOE_STORES_FROM_DATA=1; no copy) |
| `POSEDUP_RAW_DIGESTS` | the 4 raw store digests (= data_aloe_v2's own s2h-m1 DIGESTS): `c1c3b2517152741137cc4bb72dd6bdbb,09bdd43fef84574175caacf4abdddd1e,ecb1c731e8989b3f2e1b5266b51ea0fd,3b16fb8dd2660a2e8c1fe32c0d361d7b` |
| `POSEDUP_CLEAN_TARGET` | path to `data/CLEAN_TARGET.json`. `make_plan` asserts that the B side of every used item is a clean lane (train items of every epoch and the eval list). |

Asserts when RAW is on:
- **At open (every row of all 4 stores):**
  - The row map is a bijection between the kept raw rows and all sam-sep rows.
  - The map's old digests equal POSEDUP_RAW_DIGESTS, and its new digests equal the sam-sep digests.
  - Each raw store's load_node digest is pinned.
  - Each mapped raw row has the same row_key, n_frames, chunk_index, frame_start, split, speaker dirs and src_conversation as its sam-sep row.
- **On every row read:** the code arrays have the same shape and dtype, and the valid-frame mask is identical.
- **[same-as-v2] startup check:** `run_check` is the trainer's `posedup` key. It now also runs `raw_ref_check`. That builds the micro of the mapped raw row through data_aloe_v2, as a fresh module instance with its own s2h-m1 digests (26232's data path), and compares it fully with the run's 29345-path micro: B ids, depth VALUES (prefix and voice clip included), present, targets, light, cb0 VALUES, voice and P.

## Tests (CPU only; slinky-9/18/10 via srun with --mem; `tests/run_tests.sh rawid|rawchk|rawcmp|clean`; logs in tests/logs)
All runs use pose ON, POSEDUP_ITEMS=pose_ok and 426 steps. Each identity run covers the step-0 element of all 64 logical ranks (micro, singles and A wav) plus the first eval micro.
- **(a)** RAW=0 vs the original kit (`aloe-ft-32-samsep/poseduplex/code_bandmask`): 6,144 digests, 0 mismatches, **PASS**. Bandmask on: 6,144, 0 mismatches, **PASS**.
- **(b)** RAW=1 vs RAW=0: 6,144 digests. Only depth and cb0 differ, and they differ in all 384 code-value digests (64 ranks × train/singles/eval × 2). Everything else is identical, including the pose fields, own counts, light, A wav and prov. **PASS** with bandmask off and on.
- **(b) values:** `tests/raw_codes.py` checked 208 items: the step-0 item of all 64 ranks, the first eval item of all 64, 64 sampled train items across the whole 426-step plan, and 16 sampled eval items. **208/208 PASS** with bandmask off, and 208/208 with it on. Each item passed three checks:
  - raw_ref: the full equality with the 26232 v2 micro of the mapped row;
  - run_check;
  - a direct check against the raw `.npy`: B's conversation codes and cb0 labels equal the raw codes_<B side> of the mapped row.

  On average the raw codes differ from the sam-sep codes in 21.8 % of cb0 frames.
- **(c)** Plan: RAW=1 at 426 steps with pose_ok gives **d3e5e9624ef3e682** (27,156 train micros; epoch 0 b09fe689cd9e1183, 13,578 items). This is the same as RAW=0 and the original kit.
- **(d)** POSE_BAND_MASK=1 reads the bandmask dir in place. The masks and DROP_LANES apply identically at RAW 0 and 1 (the pdepth, pcb0, plight and pose_ok digests are equal), and run_check's bandmask branch passes. With the band mask on, the plan changes: 13,566 items per epoch, digest 838ce1e2dabeafee. The approved d3e5… plan is therefore the band-mask-OFF one, as in 30074.
- **Clean target:** `tools/clean_target_check.py` produced `data/CLEAN_TARGET.json` (sha256 888b71a5…).
  - All 16,214 first-chunk lanes are clean (0 flagged, 0 missing).
  - The plan's 13,894 used items (13,578 train and 316 eval) have a clean target side, 13,894/13,894, and the partner side is clean for all of them too. **PASS**.
- **Startup assert** (`tests/raw_negative.sh`):
  - The real report passes, with 13,894 items checked.
  - A report with one used lane removed fires the assert.
  - A wrong raw digest pin fires the assert.

Row map: m1n0 2948/2985, m1n1 2944/2985, m1n2 2928/2985, m1n3 2957/2985 sam-sep/raw rows, so 163 purged rows are not used.

## Env lines for the run (parent)
```
POSEDUP_RAW_CODES=1
POSEDUP_RAW_STORES=/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores
POSEDUP_RAW_DIGESTS=c1c3b2517152741137cc4bb72dd6bdbb,09bdd43fef84574175caacf4abdddd1e,ecb1c731e8989b3f2e1b5266b51ea0fd,3b16fb8dd2660a2e8c1fe32c0d361d7b
POSEDUP_CLEAN_TARGET=/data/home/amartyavarma/ppm/poseduplex-raw/data/CLEAN_TARGET.json
```
Optional trainer cosmetics (the trainer is owned by the other helper and was not touched here):
- The trainer's own `cv_` still compares only masks for depth_B and cb0 under SAMSEP. With RAW, `run_check` already covers the values.
- To make `cv_` value-based as well, use `if (SAMSEP and not A.RAW_CODES)`.
- With RAW on, the [same-as-v2] log text "CODE VALUES EXCLUDED" is stale.

data_aloe_posedup.py sha256 as tested: f228ad1c7401ce7ffa29f8507d9f05d43c0799b5ea728d2cd37f5ce205ad3cdd.

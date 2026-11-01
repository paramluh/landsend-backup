# aloe-ft-32-samsep stores: verification (10-23 22:15 PT)
**Verdict: OK.** Only the Mimi codes changed; 122 purged conversations dropped. Full numbers in `VERIFY.json`.

- **Stores:** `/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores` (m1n0..3, same file names and layout). 11,777 rows kept, 163 dropped.
- **Non-code fields** (words.arrow, persona.arrow, index.npz, rows.jsonl, rows.parquet): identical to the originals for every kept row (Arrow-level table and schema equality). The only change is that `row` / `row_in_shard` are renumbered, because the trainer asserts `row == range(R)`. T and frame windows are equal for every row.
- **Recomputed:** per-store `DONE` and `NODE.json` rows, frames, hours, split_rows and ids_blake2b (a digest over row_key + n_frames that does NOT cover codes), `SPLIT-*.json`, `eval_episodes.txt` and `PLAN.env`. New digests: 7318ffb3.. 31ecb227.. 1759200f.. 611b2e7e..
- **Codes:** every row, both lanes, equals that lane's ISOLATED stream sliced to the row window. Frames past the row's length are -1. All codes are int16 in 0..2047, T == src_frames, and no row is identical to the old codes.
  - Differ from ours: 78.9 % of all codebooks and 21.2 % of CB0. The README reports 79 % / 22 %.
- **Lanes:**
  - CB0 match with our same lane: median 0.82. With our other lane: median 0.0014.
  - 11,776/11,777 rows show the own-lane win.
  - 1 flagged row, cc3d4bbe-a355-4332-a59b-a11f269b5589/seg00/c01 (later chunk), is not a flip. Its SAM lane-0 stem is near-silent for about 600–720 s of the conversation while lane 0 is talking.
- **Trainer data path (CPU):** 1,116 micros = 158 eval + 400 train first chunks × both swaps, built with data_aloe_v2 / data_aloe_v2vox from the new and the old stores.
  - Identical in every micro: text ids, targets, light frames, P, voice-clip provenance.
  - B codes differ, and B CB0 == the new codes.
  - v2vox channel A depth is all -1, so no A Mimi is fed.
- **Decode spot-check** (Mimi fp32, 3 full first-chunk rows, wavs in `spot/`):
  - Envelope corr of the new lane with our same lane: 0.75–0.999. With our other lane: −0.49 to −0.60.
  - Own-word frames are 33–44 dB louder than other-word frames.

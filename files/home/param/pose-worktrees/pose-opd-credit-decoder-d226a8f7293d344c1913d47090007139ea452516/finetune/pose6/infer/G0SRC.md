# G0SRC: where G0COND's frame-0 features f come from at inference (pose6 workstream A2, 10-27 12:14 PT)

Code: `pose6/infer/g0src.py`. It follows `briefs/CONTRACT.md`: `g0_from_opening(opening_dir, kind) -> np.float32[8]` with `KINDS = ("openings", "openings-sam", "clip")`, and `g0_record()` returns the record plus its provenance.
Check: `python -I g0src.py --coverage [--json out]` reads every opening of every kind. It runs in 1.7 s.

## 1. Training definition (the only one used)

- **f**: `data_ppm.g0_feat`, pose5/code-811-s2p/data_ppm.py:257-275 (G0_DIM / G0_FIELDS at :243-244).
  - f = [r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w, log(H/W)], computed in float64 and stored as float32.
  - f = 0 when frame0_ok is false, a value is non-finite, or w, H or W is ≤ 0.
  - The trainer prints a WARNING and uses f = 0 (:285-296).
- **Record fields**: `fresh/fresh_pose.py:131-141, 180-182`, byte-equal in logic to `build_pose_rows_sam.py:136-146, 196-198` (PPM samdata stores).
  - Inputs come from the parquet `pose` struct. `frame0_pose` is [133,3] COCO-WholeBody (y, x, conf) in source pixels (new-pose-llm-data FORMAT.md:27,45). `orig_h`, `orig_w` are the source video's pixel size.
  - `j55 = frame0_pose[JOINTS] / [H, W, 1]` in float32, with `JOINTS = 0..12 + 91..132` (build_pose_rows_sam.py:42).
  - `frame0_root = (j55[5,:2] + j55[6,:2]) / 2`: the shoulder midpoint (COCO 5 = left shoulder, 6 = right shoulder) as (y/H, x/W).
  - `frame0_width = |j55[5,1] - j55[6,1]|`: the horizontal shoulder width / W.
  - `frame0_ok` requires all of: global_state constant and finite; frame0_pose (133,3) and finite; H > 0 and W > 0; width > 0; `allclose(global_state[0], [root, width], atol=1e-4)`.
- **Same r and w as `pose_s3q.global_features`** (lib/pose_s3q.py:153-167). The edge terms f[3:7] are its `e`.
- **g0src's copy**:
  - `g0src.g0_feat` is a byte copy. Every import asserts it equal to the training file's function text (ast segment); G0_DIM and G0_FIELDS are asserted equal too. The training file has md5 2fb643bbc0e33690e0247a3505869f52, and `G0SRC_TRAIN_DATA_PPM` overrides the path.
  - The coverage check also exec's the training text on its own and checks that the bytes equal the copy's output for all 49 sources.
  - `g0src.frame0_record` is fresh_pose's frame-0 block, with the source's stored normalised state in the role of global_state[0].

## 2. Per opening type

All values are image-normalised: y/H, x/W, and w = Δx/W.

| kind | dirs | root / width | orig_h / orig_w | frame0_ok | matches training? |
|---|---|---|---|---|---|
| `openings`, v2pure_first_window | genweb/openings/{00,01,03,08,17} | meta `sources.pose_tokens.frame0.fields` | same fields (480 x 256) | same fields | Yes. The fields are the v2 items.parquet row (`items_parquet` + `key`), built by aloe-pose build_sam_cache_v2pure.py:104-112 with the identical formula and rule. g0src checks them bit-equal to that row and to their own joints55. |
| `openings`, pkg_chunk0 | genweb/openings/{09,10,12,13,14,15,19} | `frame0.path` = demo-data/prompts/p/llm_inputs.safetensors: `frame0_pose` px + `global_state` | **not in the package**. Recovered from the recording's rows in the Aloe pose release v2 items (`orig_h`/`orig_w` columns, keyed by meta `recording_id`). | derived (training rule, global_state as gs) | Yes, recomputed from pixels with the training formula. |
| `openings`, Rosie | genweb/openings/rosie_xyvR1rjH | meta fields (root and width only) | **not in the package**. Recovered from her lane's aloe-handover-full `w00.SPEAKER_01.pose.json` `normalization.orig_h/orig_w` (1920 x 1080), found via meta store p7 / row 1525 / side user, then rows.parquet `src_dir` + `user_speaker_dir`. | derived (pose.json root0/sw0 as gs) | Yes. The pose.json recipe is the same `frame0-root-frame0-horizontal-shoulder-width-v1`, and the meta fields are checked equal to pose.json at 1e-6. |
| `openings-sam` | genweb/openings-sam/* (5 actors + 10 sam_*) | meta fields = pose.json root0_yx_normalized / shoulder_width0_normalized | meta fields `orig_h`/`orig_w`, checked equal to pose.json | derived from pose.json `frame0_pose_yxc_pixels` | Yes. The pose.json is the sibling of `sources.pose_tokens.path`. |
| `clip`, demo-data package | video/demo-data/prompts/{00..19} | gt/pose_tokens.safetensors `frame0_pose` px + `global_state` | **not in the package**. Recovered from v2 items by meta `recording_id`, as for pkg_chunk0. | derived | Yes. |
| `clip`, Rosie | genweb/people_extra/rosie_xyvR1rjH | gt/pose_tokens `global_state` only | **not stored**. Recovered from meta `row_key` → FT store row → the one lane whose pose.json state equals global_state (exactly 1 of 2 matches) → pose.json. | derived | Yes. |

### Not f sources
- The voice clip (system block) carries Mimi codes only, whether it is the person's gt/mimi_local or an uploaded clip. It has no pose and no frame 0, which is also true in training.
- The duplex playground (worker_dup, export-498) is not a G0COND model, so it is not covered here.

### Conversions, proved with numbers

**v2 / demo-data, person 09.** frame0_pose j5 = (275.9375, 232.6875), j6 = (278.28125, 24.09375) px.
- With H = 480, W = 256: root = ((275.94 + 278.28) / 2 / 480, (232.69 + 24.09) / 2 / 256) = (0.577311, 0.501526).
- w = (232.69 - 24.09) / 256 = 0.814819.
- The stored global_state is [0.5773112, 0.5015259, 0.8148193]: |derived - stored| = 0.0 in float32.
- Where 480 x 256 is confirmed:
  - All 370,978 v2 items carry it (12,497 recordings, one size each), including all 20 prompt recordings (15-63 items each).
  - The chunk store is named `...-256x480-v1`.
  - The prompts builder `gs_from_frame0` divides by 480 / 256, and its meta check `global_state_eq_derived_from_frame0` is true for 20/20.
- A wrong orig fails loud. The same clip with 3840 x 2160 gives frame0_ok false, so g0src would warn and return f = 0; it never substitutes silently.

**handover, openings-sam 06.** pose.json j5 = (2213.75, 1705.0), j6 = (2182.5, 373.75) px, H = 3840, W = 2160 (rotation -90 is already applied; the frame is portrait).
- root = (0.572428, 0.481192) and w = 0.616319, which equals the pose.json normalised values with |Δ| = 0.0.

**Overall.** Derived-vs-stored drift is exactly 0.0 for all 44 sources that have pixels. The other 5 are v2 fields, which are bit-equal to their items row.

## 3. Coverage (10-27 12:14 PT): 49/49 sources complete, 0 missing, 0 frame0 invalid (f is never 0)

| kind | sources | all fields | H x W | f[7] = log(H/W) |
|---|---|---|---|---|
| openings | 13 (12 actor openings + Rosie) | 13 | 480 x 256 (12); 1920 x 1080 (Rosie) | 0.62861 / 0.57536 |
| openings-sam | 15 | 15 | 3840 x 2160 (6); 1920 x 1080 (9) | 0.57536 |
| clip | 21 (20 demo-data + Rosie) | 21 | 480 x 256 (20); 1920 x 1080 (Rosie) | 0.62861 / 0.57536 |

- **Openings missing fields:** none. genweb/openings/_work is not an opening. The 8 persons with no opening dir (02, 04, 05, 06, 07, 11, 16, 18) use `clip`, which the deployed worker already falls back to.
- **Consistency:**
  - `openings` for the 7 pkg_chunk0 persons gives bytes equal to `clip` (the opening is package chunk 0).
  - Rosie's `openings` equals her `clip`, because handover tokens are normalised to conversation frame 0 and her clip window, frames 5585-5735, is relative to that same state.
- **Fail-loud tests**, each raising G0SourceError:
  - unknown kind;
  - v2 fields without orig_h;
  - root/width only with no row_key;
  - tampered root that no longer matches joints55 / the items row;
  - openings-sam orig that differs from pose.json;
  - clip without meta.json;
  - recording not in the v2 items;
  - missing dir.
- **f = 0 tests:** a width-0 frame0 gives ok False and f = 0. A stored state that disagrees with the pixels gives frame0_ok False, as in fresh_pose.
- **Cost:** the first v2 call takes about 1.0-1.4 s (one scan of the v2 items, then cached). Later calls take 1-3 ms, and handover sources take about 2-26 ms. A3 should warm `g0src._v2pure_orig()` when the worker starts.

## 4. Kind selection (A3's contract = the deployed worker's f0gs / sgs logic, worker.py:78-86, 334-400, 464-471)

| Case | kind | Source |
|---|---|---|
| src = start, opening has frame0 **fields**, pose prefill on | `openings` | 00, 01, 03, 08, 17 (v2 window frame 0) and Rosie |
| src = start for pkg_chunk0 persons | `clip` | deployed f0gs is None and the avatar decodes against the package; same f as `openings`, proven above |
| pose-only | `openings-sam/<p>` if present (06, 09, 10, 12, 19), else `openings/<p>`, else `clip` | |
| no-avatar person (Rosie), src = start | `openings` | |
| everything else (src = clip, no opening, prefill_s = 0, pose prefill off) | `clip` | the avatar package's frame 0 = the frame the avatar or stick render decodes against |

I agree with the contract; there are no differences.

## 5. Open points for pose6 / B2 (not decided here)

1. **Frame convention for v2-family sources.** All 12 actor openings and all 20 demo-data clips are normalised in the 480 x 256 (15:8) video frame of the Aloe pose release v2 / video chunk store, so f[7] = 0.62861.
   - The handover sources are 16:9: openings-sam, Rosie, and the FT's training items (aloe-pose-ft/stores). In a sample of 400 store lanes, 385 have a pose.json: 1920x1080 183, 3840x2160 147, 2560x1440 55, all f[7] = 0.57536. The other 15 have no pose.json.
   - So the monoplex FT never sees f[7] = 0.629, and v2 root/width are in a frame that is probably a crop of the 16:9 source.
   - g0src still gives the exact training formula in each source's own frame, which is the frame its pose tokens are normalised in. The user/main decide whether v2-family prefills are acceptable for G0COND models, or whether they should prefer openings-sam (handover, 16:9) for the 5 actors that have one.
2. **B2 should use the same frame0 rule for handover FT items:**
   - root/width are derived from pose.json `frame0_pose_yxc_pixels` with the formula above;
   - frame0_ok is true when the stored root0/sw0 match the derived values (atol 1e-4), the values are finite, and H, W, w > 0.
   - For multi-chunk rows (c01+), f should be the conversation frame 0, i.e. the pose.json normalisation frame that the tokens are relative to. That is what inference gives for the Rosie clip.
3. **Training-code pin.** If B1 copies data_ppm.py into pose6/code, point `G0SRC_TRAIN_DATA_PPM` at it as well; both are byte-asserted.

# pose6 STATUS — inference + monoplex FT with s2p cb0 + G0COND (prep only; nothing deployed, nothing launched)

Owner: tmux agent pose6 (session amartyavarma-bb). Main: amartyavarma-06 [1220bc] (since ~14:10 PT; was amartyavarma-96). Ops Board t1027-36 (main writes it). Prompt: PROMPT.md.
USER 10-27 12:03 PT: "prep inference using these mods and finetuning on monoplex again using these mods".

## Plan — 10-27 12:04 PT
Base facts: ppm811 31101 (code-811-s2p, s2p cb0 + G0COND in own groups 's2p'/'g0cond' JOIN tower_p/audio) from dcp-17850, max 19382;
exports every 360 min -> /data/home/amartyavarma/ppm/ckpt-weka; first export due ~17:40 PT.

Design decisions (fixed for every workstream; any agent that finds training contradicts one reports back, never picks silently):
- D1 SLOT, FT layout: each item's own position P_item - 1 (the system block's last position = the position just before conversation frame 0;
  811 analog: SLOT = the position before the clip's first frame). No new positions, no layout change. Packed rows: one SLOT per item.
- D2 SLOT, inference: FT models -> P - 1 inside the prefill; base 8,1,1 layout -> pose_s3q's gpos. Same position on every row.
- D3 uncond rows (audio-CFG, pose-CFG): same f as cond, unless the training code shows f dropped with the CFG stream (agent verifies).
- D4 FT optimizer groups: own groups 's2p' (JOIN tower_p) and 'g0cond' (JOIN audio), as the base run; s2p mode cb0 (base's).
- D5 FT init: tower_p.s_emb + inp.g0 loaded exactly from the base; flags on + key missing = fail loud (test-only override for
  pre-export runs with random non-zero params).
- D6 f source missing anywhere (opening / Aloe item) = fail loud; f = 0 only where training gives 0 (frame0 invalid).

Workstreams (background subagents, all in pose6/ copies):
- A1 engine: pose6/infer = pose-s2p/infer + G0COND + 811-key load path; proofs (i)-(iv) in the deployed 2-GPU layout (GPUs from main).
- A2 f sources for inference: where frame0 root/width + orig_h/orig_w live for openings, openings-sam, Rosie, clip prefill; helper + loud fail.
- A3 playground worker variant (new file, not deployed).
- B1 FT trainer: pose6/code = pose-s2p/code + G0COND port (model/data/trainer), D1/D4/D5; proofs (flags off identical, load exact,
  loss parity, grads only on pose rows).
- B2 Aloe frame-0 features for every train/eval item (27,517 + 531), from on-disk sources; sidecar if the stores lack them.
- B3 launch kit: env = aloe_mono8020.env verbatim + base + 2 flags (+ paths/tag); Rosie fixes; resolved-cfg diff; one command; ETA.

ETA: proofs + kits READY ~17:00 PT, conditional on (a) ~1 h of 2 GPUs for A1's proofs, (b) Aloe frame-0 source on disk.
Final load check on the first real 31101 export ~18:00 PT.

## Main approved plan — 10-27 12:06 PT
- GPUs: ask main ~20 min before; main frees 2 GPUs on slinky-4 for ~1 h (likely pauses the speech-only twin, restores after). Never take GPUs.
- CPU/RAM for FT proofs: check scontrol slinky-4/10/18 at the time (MemSpecLimit); ask main if nothing fits.
- SLOT P-1 OK as default; document the difference vs 31101's SLOT (PAD lane, no codes) vs Aloe P-1 (may be a system-block text token);
  state whether FT trainer and inference agree exactly there; propose a dedicated PAD position before frame 0 if cleaner, with its cost (B1).
- Uncond rows = same f: OK unless training code says otherwise; prove from code (A1).
- Agents launched 12:05 PT: A1 engine+proofs, A2 f sources (g0src.py), A3 worker variant, B1 FT trainer port+proofs, B2 Aloe frame-0
  sidecar (critical path: does a raw source exist), B3 launch kit + cfg diff. Shared rules: briefs/RULES.md.

## B2 interim + decision — 10-27 12:08 PT
- RAW frame-0 source exists: each Aloe track's pose.json (data/aloe-handover-full/DATA/<rel>/w00.SPEAKER_0N.pose.json, the file build_poseB_rows.py
  reads) has a normalization block (orig_h, orig_w, rotation, root0_yx_normalized, shoulder_width0_normalized, frame0_pose_yxc_pixels [133x3]),
  recipe frame0-root-frame0-horizontal-shoulder-width-v1 = 811's; recompute with fresh_pose.py:131-141 matches stored anchor within 1e-4.
- c>0 chunk items (9,775 / 27,517 train; 180 / 531 eval): tokens are normalised to the CUT's frame 0; raw keypoints at the item's frame 0 were
  never fetched. DECISION (pose6): (a) f = the cut-frame-0 anchor for every item. In 811, f == the anchor the tokens are relative to by
  construction; inference f = the opening's own tokenisation anchor. (b) would zero 36% of train items. Caveat documented in g0aloe/README.md.

## Contract — 10-27 12:08 PT
- A1 adopted briefs/CONTRACT.md as written (binfo["g0cond"]; G0COND + no g0f -> ValueError; SLOT P-1 in prefill on every row; step graphs unchanged). A2 confirmation pending.

## B2 DONE — 10-27 12:13 PT
- g0aloe/ sidecar: 28,048 / 28,048 items (27,517 train + 531 eval), frame0_ok false 0, missing 0. Source = each B track's pose.json
  normalization block (same tokenizer recipe as 811); g0_feat / fresh_pose frame-0 code run verbatim from code-811-s2p; anchor vs
  keypoint recompute max diff 0.0; store B pose == track token file (sha256) on all items; f digest d64a00c20f166fbb81c217ead187467b.
- Option (a): 9,775 train / 180 eval c>0 items use the conversation's frame-0 anchor (flag column anchor_is_item_frame0).
- All items portrait 16:9 -> f[7] = 0.575 constant in the FT data. 298 items rotation -90 (upright in stored frame; exact).
- Job 31117 (slinky-18, 4 CPU / 16 GB, 23 s). A3 CPU worker test = job 31118 (slinky-18).

## A2 DONE — 10-27 12:16 PT
- infer/g0src.py + G0SRC.md: g0_from_opening(dir, kind) per CONTRACT.md; all 49 playground sources covered, none missing, none f = 0.
  openings (5 actors: meta frame0 fields; 7 actors: package pixels/global_state + orig size from v2 items rows, 480x256), openings-sam (15,
  meta vs sibling pose.json, 1920x1080 / 3840x2160), Rosie (orig size from her lane pose.json 1920x1080; opening == clip f byte-identical),
  clip packages (20). g0_feat byte-equal to training text checked on import; pixel recompute == stored on 44/44; 8 fail-loud cases raise.
- DECISION (pose6): f = the frame the pose is rendered against (contract), so actor openings / clip packages give f[7] = 0.629 (480x256)
  while the FT data is all 16:9 (0.575). Accepted, formula exact in each source's frame; caveat: FT never sees 0.629 (base 811 saw varied
  aspects). openings-sam not preferred on aspect grounds.

## A3 DONE (CPU) — 10-27 12:17 PT
- worker/worker_s2pg0.py = deployed genweb worker.py + marked pose6 lines (worker.diff 182 lines): pose6 engine + g0src by abs path; refuses
  ckpts without s2p cb0 + inp.g0 or with zero-filled new params; GENWEB_CK required; g0f per run from g0_from_opening (deployed worker's own
  frame-0 choice), optional request override (8 finite), fail per run before engine start; warm 39 sources at start; weights id s2pg0
  (worker id s2pgw<job>: reaper parses first 5 digits as job id).
- CPU test 9/9 PASS (job 31119): params/labels/schema identical to deployed, no deployed-engine import, 126 routing cases, overrides, errors,
  real g0src 126/126 finite, ckpt check, warm 39/39 1.67 s.
- Hot-swap (NOT done; main's word only): Rosie pattern, dedicated worker unit s2pg0_unit.sh, hidden verify run, server.py allow-list via
  genweb DEPLOY, picker option, changelog line; rollback = grey option, cancel idle job.
- Open: GPU test (2 GPUs, -c 8, 120G, ~15 min) -> run in A1's GPU window. Worker is for FT-layout models (base 811 export would put f at
  P-1 in a layout the base never trained on; base layout SLOT = pose_s3q gpos, A1 reports).

## OPEN DECISION (user via main): decp_bb link in the FT — 10-27 12:24 PT
- 31101 runs D33 decp_bb link OFF (prod_811_fresh_append5_s2p.env PPM_DECP_BB_OFF=1 since 10-20; exports decp_bb_alpha 0 + decp_bb_off 1).
  mono8020 verbatim trains it ON (QQA_DECP_BB_ALPHA=1.0, QQA_LR_DECP_BB=1e-5): its base local-11488 still had the link on.
- pose6 recommends OFF (FT step 0 = base exactly; inference alpha 0). Sent to main 12:2x for the user. Both built; kit default OFF pending.
- B1: porting QQA_DECP_BB_OFF (811 PPM_DECP_BB_OFF semantics; unset = byte-identical) + proof. B3: kit line + --link-on; scan the env for
  other knobs that assumed the old base's state. A1: verify engine alpha 0 for 31101 / FT exports.
- B1 status 12:23: G0COND port + own groups + init (local shards or consolidated DCP w-N) coded, wired to g0aloe; CPU proofs job 31121
  (slinky-4, 2 CPU / 16G), ETA 13:45. A1 CPU checks job 31120 (slinky-18, 4 CPU / 48G).

## B3 env scan — 10-27 12:26 PT
- Only mismatch vs 31101 state = decp_bb link (open decision above). Same: CD 2, attn32, dec K 8, replicate, no pose head, key set (only the
  2 new keys new; w-15973 / export-357 = FT key set), pose normalisation recipe. SLOT content differs (811 PAD vs FT P-1 '\n'; D1).
- Kit: QQA_DECP_BB_OFF=1 explicit line (default OFF pending user) + launch.sh --link-on; CFG_DIFF flags it.
- Dry cfg (trainer's own code): world 64 vs 24272 = paths + init + s2p/g0cond keys only. World 8 adds micro_layout lpt + save_every_min
  -> B3 to give both kits (64-rank verbatim, 8-GPU default) with neutrality proofs / flags, memory fit (optoff), ETA.
- B2 resumed: Mimi recipe check (31101 pose rows vs Aloe stores vs live engine) -> g0aloe/MIMI_RECIPES.md (s2p reads B's cb0).

## A1 CPU PASS + GPU request — 10-27 12:28 PT
- CPU checks PASS (jobs 31120, 31123): load path both formats (local shards + consolidated DCP), exact tensors; fail-loud cases; real key
  sets (31101 export adds exactly inp.g0 [5120,8] + tower_p.s_emb [2048,2560]); SLOT = P-1 = '\n' on all 528 Aloe eval items checked.
- LINK OFF: pose6 engine forces alpha 0 when the ckpt records decp_bb_off (synthetic: 31101 DCP side; alpha 1.0 + off; local cfg alpha
  1.0 + off): slot 0 == proj(hv_p) bitwise. BUG found: deployed + pose-s2p engines ignore decp_bb_off (31101 saves would run alpha 1.0)
  -> reported to main (not touched).
- B1 told: FT export (train_aloe_pose.py:665 hardcodes decp_bb_alpha 1.0) must record alpha 0 + decp_bb_off 1 under link-off.
- GPU request to main: 2 GPUs one node, 4-8 CPU, 128G, ~80 min (A1 infer/tests/test_engine6.sbatch <=65 min, then A3 worker GPU test).
- 12:28 PT main: frees speech twin's 2 GPUs on slinky-4 (~80 min); WAIT for "GO slinky-4"; sbatch -w slinky-4 --gres=gpu:2 -c 8 --mem=128G;
  never GPUs 6-7 (throttled) -> A1 adds a GPU/clock guard; tell main when done to restore the twin.
- main: prepare (NOT deploy) decp_bb_off fix patch for deployed aloe-pose-monoplex-infer fast_ppm.py + pose-s2p, proof that export-357,
  Rosie export-18, duplex export-498 are byte-identical (none record decp_bb_off); goes in with the first model swap on user's OK -> A1,
  pose6/enginefix/.

## Disk — 10-27 12:30 PT
- ALERT to main: slinky-18 /scratch 100% (15 GB free / 1,760) while motasr2 runs there. pose6 share = 2 GB kit-test copies
  (/scratch/amartyavarma/pose6-kit-test, job 31122: w-15973 shard subset copies + synthetic DCP); deletion awaits main's OK.
- 8-GPU FT kit needs ~564 GB node /scratch for the recipe's 6 exports (88.4 GB each at world 8); 64-rank kit 237 GB/node (mono8020 rule).
  Default = recipe exports + fail-loud preflight; --final-export-only (~118 GB) opt-in = user call (export schedule only).
- A1 GPU guard in test_engine6.sbatch (exit 3 on slinky-4 GPU 6/7 or SM clock < 75% max under burn); waiting for main's "GO slinky-4".
- 12:30 PT: main OK -> deleted slinky-18 /scratch/amartyavarma/pose6-kit-test (1.2 GB). Asked OK for pose6-a1 synthetic 3.9 GB. slinky-18 /scratch still 100% (16G).
- 12:31 PT: main OK -> deleted slinky-18 /scratch/amartyavarma/pose6-a1 (3.9 GB synthetic). pose6 now has nothing on slinky-18 /scratch (20G free, 99%).
- 12:31 PT B1: QQA_DECP_BB_OFF ported (export train_cfg.json + meta side record alpha 0 + decp_bb_off 1; unset = mono8020 bytes); init proof (stage D) PASS; remaining stages in job 31133 (slinky-4). B1 test data only slinky-4:/scratch/amartyavarma/pose6-b1-tests (<1 GB).

## GPU window (main GO slinky-4, twin GPUs 4-5 freed 12:31 PT) — 10-27 12:32 PT
- A1 engine proofs = job 31138 (slinky-4, 2 GPU, 8 CPU, 128G, <=65 min), RUNNING 12:32 PT. Triton cache switched to pose6's own
  /scratch/amartyavarma/pose6/triton-cache (was genweb's). A3 worker GPU test: overlay test-only (export-357 + seeded non-zero new params),
  to be chained afterany:31138. Tell main the moment both finish (twin restore).
- 12:33 PT: 31138 exited 3 at the guard 12:33 (false trip on healthy GPUs 4-5: lazy 2-thread context init + 75 % clock threshold wrong;
  GPU 4 684 TFLOPS at 1395 MHz = normal power cap). Guard fixed (>= 450 TFLOPS, median SM >= 50 % max, no 0x20/0x40/0x80). Resubmitted
  as job 31140 (12:33 PT).
- enginefix READY, NOT deployed (pose6/enginefix/README.md): deployed monoplex fast_ppm.py + pose-s2p honour decp_bb_off; test_fix PASS
  (job 31136): export-357, Rosie export-18, duplex export-498 identical ckpt_info + alpha 1.0 before/after; none record decp_bb_off.
  Duplex live_ppm_pose.py has the same bug (lines 198, 215), NOT patched -> told main.

## B3 DONE (launch kit, NOT submitted) — 10-27 12:34 PT
- ONE command: bash /data/home/amartyavarma/ppm/pose6/kit/launch.sh --base /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-<step> [--go]
  (dry run default; --go refused unless base keys/shapes, sidecar coverage, node idle, /scratch free all pass; rechecked in-job).
  --nodes 1 (default, 8 GPU, world 8, Rosie AdamW offload, ~14 h [12-16], no mid-run saves) | --nodes 8 (64-rank verbatim, ~1.5 h;
  24272 = 80.2 min, 11.6 s/step). Base read in place from DCP w-N (no conversion); --stage = bit-exact conversion (0 tensors differ).
- CFG_DIFF vs 24272 (trainer's own code; mono8020 code+env reproduce 24272 with 0 diffs): 64-rank = 16 keys (names/paths, base,
  s2p/g0cond + own-group LR/clip, flagged decp_bb_off); plan + rows digests == 24272's. 8-GPU adds micro_layout row_per_rank->lpt,
  save_every_min 60->100000. Flags off = 8 keys (names + base).
- 8-GPU not bit-identical (decoder nanostep frame split draw + fp sum order); normalisers/dummies/code path unchanged.
- Rosie fixes: unloaded-keys assert relaxed (31101 has no pose_head keys; both kits); optoff (8-GPU; byte-identical to job 28249's
  checked copy); skip final full save (8-GPU); eval pad (inert).
- Preflight: w-15973 / export-357 fail exactly on the 2 missing new keys; sidecar 27,517/27,517 + 531/531. Positive check pending the first
  31101 export (~17:40 PT).
- USER DECISIONS: (1) decp_bb link OFF (default, recommended) vs --link-on; (2) 8-GPU (~14 h, not bit-identical, no mid-run resume) vs
  64-rank (~1.5 h, 8 nodes); (3) /scratch: 8-GPU needs 528 GiB (only slinky-15 at 31101 hand-back fits; slinky-10 maybe), 64-rank 237
  GiB/node; --final-export-only cuts to 113 GiB (export schedule only).
- 12:34 PT main: prepare the same decp_bb_off patch for aloe-posedup-live/live_ppm_pose.py (NOT deploy; same proof incl. export-498) -> A1, pose6/enginefix/duplex/.

## Main decisions on the kit — 10-27 12:34 PT
- (2) DEFAULT = recipe VERBATIM, 8 nodes x 8 GPUs (copy-reference-exactly), on 8 of the 16 production hosts after 31101 hands them back
  (or at a 31101 pause if the user wants an earlier base), before the MoT resume. 1-node variant = documented fallback only.
- (3) keep all 6 exports; disk check 237 GiB/node vs pose5's hand-back state. (1) decp_bb: with the user; default OFF.
- Base = 31101's newest COMPLETE export. B3 retargeting (host list, live disk check, MoT >= 64 GiB hand-back interplay, --base newest).

## SEQUENCING (FT)
1. 31101 ends at step 19382 (~28 h from 11:41 PT 10-27) or the USER asks for a pause (save + export) for an earlier base.
2. FT 64-rank on 8 of the 16 hosts, base = newest 31101 export (~1.5 h incl. evals; export-357 consolidated to /data). Trigger: user's go via main.
3. Hand-back: FT node-local exports HF-backed (private repo); any deletion only on main's OK; hosts back to >= 64 GiB free.
4. MoT resume (motguard / main).
- 12:35 PT: A3 worker GPU test = job 31142 (slinky-4, afterany:31140; export-357 + test-only overlay, worker_s2pg0.py md5 asserted unchanged).

## B2 Mimi recipe check DONE — 10-27 12:36 PT (g0aloe/MIMI_RECIPES.md)
- Same: Mimi weights (kyutai/mimi @89091b3e sha bac7e850...), 24 kHz, 10 s local window. Different: loudness scope (31101 per segment vs
  Aloe whole track; 28% Aloe tracks > 0.5 dB short of -23), encoder state (fresh per segment vs continuous 30 s-chunk stream), precision
  (31101 maybe fp16 vs fp32). npl2/npl3 unchecked (node-local only).
- cb0 histograms (speech frames): plan1 vs npl5 0.0019 bits (= floor); fresh vs Aloe 0.084 bits (~ old-vs-new recipe size, but content
  differs too; unresolvable without audio). Unmasked 0.286 bits, dominated by listening silence: ~29% of Aloe B frames vs 0.4% of npl5
  -> s_emb barely trained on silence codes (coverage gap; the FT trains s_emb on Aloe codes).
- pose6 view: no action for the FT (recipe = Aloe stores verbatim; s_emb trains in the FT; at inference B's codes are model-generated,
  i.e. FT-distribution). No download. FLAG for the user (pre-existing, not ours to change): deployed engine voice clips skip the -23 LUFS gain.
- 12:39 PT A1: duplex enginefix READY, NOT deployed (enginefix/duplex/live_ppm_pose.py + .patch). One helper in all 3 engines: alpha 0 iff
  decp_bb_off in the record or its nested cfg (FT exports may carry it only in cfg). Each patch = 2 alpha lines removed + helper. Every
  configured duplex model (export-498-dcp slinky-18 /scratch serving copy + /data copy, same md5; pdsamples/export-498), export-357, Rosie
  export-18: identical ckpt_info, alpha 1.0 before/after, no decp_bb_off. Synthetic off records 1 -> 0. Jobs 31143/31145/31146 PASS.
  Unchecked: CK_LABELS fallbacks export-333 + slinky-10 /scratch export-498 (slinky-10 only, no free CPU).
- 31140: guard PASS (682 / 690 TFLOPS); (i)/(ii) eager/free PASS bit-identical incl. orig-vs-orig floor; continuing.

## B3 kit retargeted (64-rank default) — 10-27 12:41 PT
- Default: launch.sh --base newest = mono8020 64-rank verbatim; --nodes 1 = fallback only. Default --hosts (pose5 hand-back projection,
  11:46 PT records): slinky-15,14,5,7,1,2,17,3 (563 -> 95 GiB). --go refuses until live df passes.
- DISK BLOCKER: even after every pose5 hand-back item + final dcp-19382 are removed, only s15/s14/s7 reach 237 GiB (5th best 229).
  --saves (user call; save cadence only, maths unchanged): hourly (recipe) 237 GiB -> 3 hosts; final (final FULL save only) 175 GiB -> 7
  hosts (s3 2.7 short); none (6 exports kept, no resume point) 113 GiB -> all 8 once pose5 tiers freed on s1/s2/s17/s3.
- Base: 31101 exports every 360 min only (no export at 19382): last ~11:40 PT 10-28 (~step 19,100-19,150); run ends ~16:30 PT 10-28.
  For an end-of-run base: main converts dcp-19382 with ckpt_local_export.sh -> local-19382 (~84 GB /data); 'newest' finds it.
- FT node-local files under /scratch/amartyavarma/pose6-ft/; before MoT resume: HF backup (proposed private InternalCan/ppm-pose6-ft-s2pg0)
  -> main OKs removal -> pose5 handback_free64.py -> MoT resume on main's GO. Kit prints, never deletes.
- Job 31147 (slinky-4): mono8020 code still == 24272 cfg (0 diffs); 64-rank kit vs 24272 = 16 keys (7 names/paths + base; s2p, lr.s2p,
  clip.s2p; g0cond, g0_sidecar, lr.g0cond, clip.g0cond; flagged decp_bb_off); plan digest a5d995bcec45dc6f + rows 3681cc26ea97cad7 == 24272.
- pose6 recommendation: --saves final (a full save at the end stays possible) + main frees >= 3 GiB of regenerable caches on slinky-3 (or
  swaps in another host); base = converted dcp-19382 (end of run) unless the user wants an earlier base.

## Main APPROVED — 10-27 12:42 PT
- (1) --saves final (keep the full final save: masters + AdamW). slinky-3 caches are MoT's (off-limits) -> swap slinky-3 for another
  production host that fits per pose5's hand-back projection (B3).
- (2) End-of-run base: convert dcp-19382 with ckpt_local_export.sh -> local-19382 (~84 GB /data), own verify, /data ledger >= 300 GB after.
  ALTERNATIVE (if the user wants to start earlier): the last 6-hourly 31101 export (~step 19,100, ~11:40 PT 10-28), no conversion.

## SEQUENCING (FT) v2 (supersedes the list above)
1. 31101 ends at step 19382 (~16:30 PT 10-28) [or the user asks for a pause/earlier base -> newest 6-hourly export].
2. main triggers: ckpt_local_export.sh dcp-19382 -> local-19382 on /data (~84 GB) + its verify + kit preflight keys/shapes; /data >= 300 GB after.
3. user's go via main: kit launch.sh --base newest (= local-19382), 64-rank on 8 hosts, saves final, ~1.5 h.
4. hand-back: FT node-local -> HF private backup -> main OKs removal -> pose5 handback_free64.py (>= 64 GiB/host).
5. MoT resume (motguard / main).
- 12:42 PT B3: no host replaces slinky-3 under the 175 GiB rule (others 137-168 GiB after all 31101 tiers + final dcp-19382 removed). Actual
  write per host (saves final) ~135 GiB (full save ~73 + 6 export slices ~62; rank-0 host 148). Asked main: need = write + 30 GiB floor
  (pose5's 31101 floor) = 165 (rank 0 178 on slinky-15) -> slinky-3 (172.3) fits as-is, nothing of MoT's touched; slinky-8 (167.7) spare.
  B3 implementing as default pending main's OK (recipe rule kept as --disk-rule recipe).
- 12:43 PT main APPROVED: disk need/host = measured FT write (~135 GiB; 148 rank 0) + 30 GiB floor = 165 GiB (178 rank 0 on slinky-15);
  slinky-3 stays, slinky-8 spare, nothing of MoT touched. Preflight measures real free space at launch and refuses below need; the run logs
  the write per host at the end.
- 12:43 PT SUBAGENT SESSION LIMIT (HTTP 429) until 21:00 UTC = 14:00 PT: B3 (mid disk-rule edit) and B1 (waiting on its comparison, job 31133)
  stopped; work saved; resume both via SendMessage after 14:00 PT. Slurm jobs keep running: 31140 (A1 GPU), 31142 (A3 GPU, afterany),
  31133 (B1 CPU). pose6 watches 31140/31142 itself and tells main when the GPUs are free.

## A1 GPU proofs (job 31140, slinky-4 GPUs 4-5, 12:33-13:01 PT, 27:45) — 10-27 13:02 PT (infer/tests/out/report_31140.md)
- (i) flags off vs DEPLOYED engine and (ii) zero new params (s2p cb0 SE=0; s2p cb0 + G0COND SE=0, G0=0, f=F_A): bit-identical, incl. the
  orig-vs-orig floor, eager + graphs, free / free_cfg / poseonly (50-150 frames; B lane, speech/pose codes, hv_p, pose cb0 logits both rows,
  hv_s, speech cb0 logits, hn both rows). PASS.
- (w) wiring PASS: prefill K/V differ only at SLOT 133 (P 134) in both rows; uncond SLOT K/V == cond (same f); f = 0 == off bit-identical;
  F_A vs F_B changes outputs; fail-loud on new-without-g0f, off-with-g0f, s2p-only-with-g0f, SLOT not '\n'.
  Non-zero scales: inp.g0 N(0, 0.06^2) (||g0(F_A)|| 6.77 ~ median per-frame ||inp.pose|| 6.79); s_emb N(0, (0.02/sqrt32)^2).
- (iii) engine vs TRAINING forward (pose6/code model, B1 port), teacher-forced 300 frames, 4 prompt types (clip P70, persona P929, empty P5,
  clip+persona P212), cond + pose-CFG uncond rows: text / speech cb0 / pose cb0 (both rows) / hv_p / zb / pose decoder all at the
  zero-params floor. ONE OUTLIER: hn UNCOND max abs on the clip micro = 6.5 graphs / 5.25 eager (floor 1.0; ref absmax 109.5; > the
  trainer's own mod effect 2.12), while pose cb0 uncond logits stay at floor (0.125). Other micros: uncond hn 1-1.5. -> A1 to localise
  (position/frame/channel, uncond reference construction) after the 14:00 PT limit reset. Not READY until explained.
- (iv) latency (graphs, sync median): today 60.96 | s2p cb0 66.18 | s2p cb0 + G0COND 66.31 ms/frame (free); free_cfg 64.99 | 69.99 | 70.06;
  poseonly 61.2 | 66.35 | 66.47. G0COND adds ~0.1 ms (prefill only). Within the 80 ms budget.
- A3 worker GPU test 31142 running on the same GPUs since 13:01 PT (guard passed).
- 13:03 PT main: keep slinky-4 GPUs 4-5 for A1 (iii-strong + outlier localisation), chained right after 31142, until ~14:00 PT at the latest; then tell main to restore the twin.
- 13:03 PT A1 final (stage 1): engine README/TESTS done; base 8,1,1 layout never served by this engine/playground (start() needs P >= 5;
  genweb serves FT ckpts only). HAZARD: deployed + pose-s2p engines silently drop inp.g0 / s_emb keys -> A1 adds fail-loud to the
  enginefix patches (not deployed). Next: iii-strong + outlier localisation, chained after 31142 (window until ~14:00 PT).

## A3 worker GPU test PASS — 10-27 13:04 PT (job 31142, 13:01-13:03 PT, export-357 + test-only overlay)
- 8/8 PASS in 92.8 s: W0 worker up with s_emb cb0 + inp.g0 non-zero; R1 openings (Person 01), R2 clip (09), R3 openings-sam (06), R4 Rosie
  opening: each done, summary g0f == g0src(dir, kind); R5 override = own f -> bit-identical; R6 f = 0 -> 49/50 pose frames differ; E1 engine
  refuses start without g0f. worker_s2pg0.py md5 unchanged.
- GPU hold: placeholder job 31150 (slinky-4, gpu:2, 8 CPU, 128G, <= 55 min) keeps GPUs 4-5 between jobs; A1 iii-strong runs inside via
  srun --overlap; scancel 31150 when done -> tell main (twin restore), by ~14:00 PT.

## A1 iii-strong + outlier localised; GPUs released — 10-27 13:11 PT
- iiis (13:04-13:08) + loc2 (13:08-13:10) inside hold 31150; hold scancelled 13:11 PT; main told to restore the twin.
- iii-strong (clip + empty micros, 150 frames, graphs, cond + uncond; g0 sigma x10 -> ||g0(f)|| 67.7, s_emb x30): effect vs parity max abs
  text 16.3-16.8 vs 0.5-1.0; speech cb0 8.6-9.8 vs 0.14-0.37; pose cb0 cond 3.3-5.6 vs 0.125; uncond 4.9-5.4 vs 0.125; hn 7.8-57 vs
  0.56-1.0 (ratios 14-60x). Parity TVs == the zero-param floor run. (iii) PASS.
- Outlier: uncond row, frame 1 (pos P+1 = 71), channel 1660: ref -34.25 vs engine -39.5; floor diff there 0.0; that frame's hn absmax 40.75
  (~12 elsewhere). Does NOT grow with scale (strong: uncond hn max 1.0). Ill-conditioned: trainer itself moves 2.81 there for a 1-ulp f[0]
  change; valid bf16 paths disagree by 1.5-5.25 (step vs trainer, FA3 prefill vs trainer, step vs prefill). Uncond reference verified:
  silence at the 61 clip positions + every conversation position >= P+2 == engine rule sil_t[T-P-CD]; A, ids, present, pdepth unchanged;
  g0 at the same SLOT with the same f. Accepted as bf16 conditioning, not a mismatch.
- enginefix extended (NOT deployed): deployed / pose-s2p / duplex build_model refuse inp.g0 / s_emb ckpts (pose-s2p: inp.g0 only), naming
  the pose6 engine; jobs 31151 + 31152 PASS (export-357, Rosie export-18, both export-498 copies pass; synthetic keyed ckpts raise).
- DELIVERABLE A status: engine + g0src + worker + proofs (i)(ii)(iii)(iv)(w) DONE. Open: final load check + worker re-run without overlay
  on the first real 31101 export (~17:40 PT).

## B1 proof results (job 31133, slinky-4 CPU, finished 12:50 PT; read by pose6 while B1 is rate-limited) — 10-27 13:11 PT
- (1) mono8020 trainer vs pose6 flags off: fwd, all grads, decoders after nanostep, after optimizer step, grad norms, loss, stats, resolved
  cfg: identical (cfg diff []). PASS. Micros: flags off == mono8020's; flags on minus g0 keys == off; g0 keys only when on.
- (5) zero new params vs off: fwd/grads/after-step identical on every shared param; only the 2 new params extra; cfg diff = expected keys;
  JOIN groups ok. PASS.
- (4) grad structure: s_emb grad rows == cb0 codes present (1650/1650); g0 grad == sum dy f at the SLOT rows (packed rows: g0pos [4, 229,
  8516] and [211], one per item); rel err 2e-3 (bf16). PASS.
- non-zero new params change fwd; LINK OFF (QQA_DECP_BB_OFF=1): zb_p not computed, decp_bb no grad + unchanged after step, loss equal to the
  link-on zero-alpha path here; exports: off == mono8020 train_cfg/meta (minus save_time); link-off sides record alpha 0 + decp_bb_off 1. PASS.
- (C) per-row-type parity vs the BASE's own trainer (code-811-s2p model forward, same weights incl. non-zero s_emb/g0): Aloe packed (3 SLOTs),
  single empty / clip / persona / both, and a TikTok 811 pose micro: all terms bit-identical (max abs 0.0), 128 grads identical (0.0). Key
  set FT == 811 (2,152 keys; new = inp.g0 [5120,8], s_emb [2048,2560]); w15973 == FT-off key set.
- (D) init: base DCP -> flags-on model across world 3 -> 2: 128/128 tensors exact, new params exact, unloaded [], random_fill []. PASS.
- Pending on B1 resume (14:00 PT): TESTS.md write-up; explicit statement that every mono8020 micro is a pose micro (so new-param grads
  only on pose rows holds trivially in the FT) with the count check.
- 14:01 PT: limit reset; B1 (TESTS.md, pose-rows-only count, SLOT comparison README, md5s) and B3 (approved disk rule, saves final, local-19382 step, dry_all + CFG_DIFF refresh) resumed. A1 + A2 + A3 + B2 DONE.

## B1 DONE — 10-27 14:03 PT (pose6/code: README.md, TESTS.md, CODE_MD5.txt)
- All PASS (job 31133): flags off == mono8020 trainer bit-identical (cfg 0 diffs, fwd, loss, all grads, decoders after nanosteps, clip norms,
  params after AdamW, export files, micros); init exact from w-N DCP and local shards, world 3 -> 2 (128/128 incl. new keys; missing keys
  or flags-off-with-keys fail loud); parity with 31101's model (non-zero new params) on Aloe packed / 4 prompt types / TikTok 811: max abs
  0.0 fwd + grads; every FT micro is a pose micro (22,848 train incl. 23 dummies + 529 eval, from mono8020's own log) -> new params get
  grads only on pose rows; zero new params == off incl. after step (own groups); link-off matches the base's path, exports record alpha 0 +
  decp_bb_off 1; g0f == B2 sidecar byte-exact on 28 real micros.
- SLOT: P-1 = footer '\n' (id 198) on channel B for every prompt type, no codes / pose / text target; first pose label at SLOT+3 = the
  base's offset. Base SLOT = PAD segment start. Dedicated PAD alternative: shifts every position, likely changes the rows digest, +1 at
  inference -> not recommended.
- B1's "engine local-copy alpha" concern checked by pose6: stale. pose6 engine line 171 wraps it in _link_alpha(path, meta['cfg'], ..),
  which honours decp_bb_off (the FT trainer writes CFG['decp_bb_off'] = 1). The deployed engines' bug is covered by enginefix.
- B1 test data: slinky-4:/scratch/amartyavarma/pose6-b1-tests (~2 GB synthetic) kept; deletion only on main's OK.

## B3 DONE (final kit) — 10-27 14:08 PT (pose6/kit/README.md, CFG_DIFF.txt)
- Default: 64 ranks on slinky-15,14,5,7,1,2,17,3 (rank 0 pinned slinky-15; placement only), spare slinky-8, never slinky-9; --saves final
  (FULL save at step 357 only); disk need = measured write + 30 GiB = 165 GiB (178 rank 0) via live df, launcher + in-job; actual write per
  host logged to logs/disk-<job>.txt. --disk-rule recipe (175) kept.
- At hand-back (projection): s15 563, s14 208 fit as-is; s5 needs pose5's npl5 tier (~11 GiB); s7/s1/s2 need pose5's tiers incl. the
  dcp-17850 pin; s17/s3 need pose5's tiers + dcp-19382 (after conversion + HF verify) -> 177.6 / 172.3. Nothing of MoT's touched.
- Base local-19382: convert_final.sh prints; main runs sbatch -w <31101's 16 hosts> convert_final.sbatch (copy of ckpt_local_export.sh +
  --host-as for logical hosts s15->slinky-9 / s7->slinky-10, as monoft did for dcp-17850); verify = converter --finalize (128 rank files +
  rep.pt) + kit preflight (keys/shapes/bf16/finite + sidecar). --base newest then picks local-19382.
- /data LEDGER: 657 GB free now; 31101 has PPM_EXPORT_KEEP=0 (keeps all exports) -> ~303 GB at its end; the 89 GB conversion would breach
  300 -> main must prune superseded (HF-verified) w-N exports first. 31101 end now projected ~17:10 PT 10-28.
- Job 31155: mono8020 code == 24272 (0 keys); default kit = 17 keys (7 names/paths, base, 7 s2p/g0cond incl. sidecar path, flagged
  decp_bb_off, save_every_min 60->100000); all mods off + hourly = 8 keys; plan + rows digests == 24272's. B1's code md5s pinned.

## READY — 10-27 14:08 PT
- A (inference): pose6/infer engine (s2p cb0 + G0COND, link-off fix) + g0src (49 sources) + worker variant (pose6/worker) + proofs
  (i)(ii) bit-identical, (w), (iii)+(iii-strong) at floor, (iv) 66.3 ms/frame; enginefix patches for 3 deployed engines (NOT deployed).
- B (FT): pose6/code (G0COND port + own groups + exact init + link-off knob) proven; g0aloe sidecar 28,048/28,048; kit 64-rank verbatim.
- Pending, automatic: first real 31101 export (~17:40 PT; watcher running) -> engine key/load check, kit preflight --base newest (dry),
  then worker GPU re-run without overlay (needs 2 GPUs ~15 min from main).
- Decisions: USER decp_bb link (default OFF). MAIN at hand-back: free pose5 tiers on s5/s7/s1/s2/s17/s3 (+dcp-19382 after verify on s17/s3);
  prune superseded w-N so /data stays >= 300 GB after local-19382.
- 14:11 PT: main restarted -> amartyavarma-06 [1220bc]; READY re-sent there.

## URGENT (main amartyavarma-06, 14:29 PT): mono17850 — recorded 10-27 14:31 PT
- User found: playground monoplex export-357 = FT from local-11488; the 10-22 "monoplex FT from dcp-17850" was mis-briefed as Xabi v2pure
  (memory monoplex-means-aloe-mono-rows.md). NEW KIT mono17850 = aloe-pose-mono-8020 (24272) EXACTLY, init = 8,1,1 dcp-17850 (old arch: S2P
  off, G0COND off; pose6 kit with flags off certified == mono8020 trainer); decp_bb line explicit (default OFF, pending user).
- C1 (new agent): dcp-17850 -> /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850 (ppm-local-bf16-shards-v1, world 128, like
  local-17641). Weka copy gone (pose5/dcp17850-weka empty); node-local copies only on 31101 hosts (off-limits) -> HF InternalCan/ppm-v1-c
  range reads of model tensors only; verify converter on dcp-17641 vs existing local-17641 (byte-identical subset) + kit preflight. slinky-18
  4 CPUs, stream to /data. ETA ~16:30 PT.
- B3 (resumed): pose6/mono17850/ kit, options (a) 64 ranks on 8 of 31101's hosts after STOP_AFTER_SAVE (per-host /scratch, 31101
  resume-back, total delay; first est ~1h50m) and (b) 8-GPU on one spare node (displaces motasr2; ~14 h); CFG_DIFF = init + paths/tag +
  decp_bb. NOT launched, nothing paused. Plan + ETA sent to main 14:3x; READY ETA ~16:45 PT.
- /data ledger: local-17850 +84 GB now; 31101 exports (KEEP=0) -> ~219 GB at run end unless superseded w-N pruned (main).
- 14:31 PT main: mono17850 plan APPROVED as written; (a) + (b) both go to the user at READY; any w-N / ckpt prune needs main's explicit go (list candidates only); one-line ping when local-17850 verified.
- 14:38 PT B3 mono17850 kit (pose6/mono17850/{README.md, CFG_DIFF.txt, launch.sh}): cfg vs 24272 = 9 keys (init local-17850 [stand-in until
  C1], 7 names/paths, decp_bb_off); plan + rows digests == 24272; preflight passes on local-17641 + export-357 (2150/2150 keys).
  (a) 64 ranks on s15(r0),14,5,7,1,2,17,3 (spare s8) during a 31101 pause: /scratch (hold-chain 14:31) 570/215/166/136/116/112/107/102 GiB
  (-43 while the previous save is held); recipe hourly saves need 238 -> only s15; --saves none needs 92 (105 r0) -> all 8 but only after
  the pause save is HF-verified (47-96 min). Final export ~2.2-3.0 h after trigger; then HF backup of FT node-local (~62 GiB/host, 35-65 min)
  + main OK removal before 31101 resumes (needs ~71 GiB; s3 would have 40); resume via pose5 relaunch (auto, same hosts, ~6 min; roller
  31105 keeps running). 31101 delay ~2.9-4.2 h (~200 steps).
  (b) --nodes 1 --hosts slinky-10 [--final-export-only]: displaces motasr2 on s10; 528 GiB (or 113 with final-export-only); ~14 h; world 8 lpt
  not bit-identical (user OK). Asked B3 for (a') = (a) + --final-export-only numbers.
- 14:41 PT B3 (a') = (a) + --saves none --final-export-only: FT starts right after the pause save (no HF-verify wait); final export ~1.4 h after
  the pause trigger; 31101 delay ~1.5 h (~78 steps). Need 41 GiB/host (44 r0 s15): all 8 clear it even with the previous save held (lowest s3
  59 held / 102 free). Nothing to back up/remove before 31101 resumes (FT leaves ~11 GiB/host; by 31101's first new save at resume+60 min
  the pause save is HF-verified -> s3 ~91 vs ~71 needed; risk only if HF verify > ~140 min). cfg vs 24272 = 11 keys (init, 7 paths,
  decp_bb_off flagged, save_every_min, export_steps -> []). Dry run 31171. /data 1,143 GB free -> ~878 after local-17850 + 31101 17:40
  export + FT export. Command: bash /data/home/amartyavarma/ppm/pose6/mono17850/launch.sh --saves none --final-export-only --go.

## GO mono17850 option (a') — main 14:41 PT; recorded 10-27 14:42 PT
- Auto-launch the moment local-17850 is verified; 64 ranks on 8 of 31101's hosts (s15 r0,14,5,7,1,2,17,3; spare s8); --saves none
  --final-export-only; decp_bb link ON (= mono8020, recipe verbatim; = the 17850 Xabi FT, the best). Pause coordinated with pose5
  (amartyavarma-39): STOP_AFTER_SAVE on 31101 with watchdog v2 / roll supervisor / hold chain told it is intentional. Resume 31101 automatically
  right after the FT's final export is written + verified. Push the final export to private HF (as the 27757 chain). Pings to main: launch,
  final export verified, 31101 resumed.
- 14:42 PT: pose5 asked to confirm the pause/resume mechanics (pose5 owns every 31101 action; RESUME marker pose6/mono17850/RESUME_31101 as
  fallback). B3 preparing launch cmd (--link-on), verify_final.sh, push_hf.sh.
- 14:43 PT main 14:43 "start NOW" (pause 31101 now + node-local conversion) was RETRACTED by main within the minute (misread the user). pose6 had
  sent pose5 "PAUSE NOW" + B3 the node-local conversion; both CANCELLED at once. Checked 14:43:08: no STOP_AFTER_SAVE / SAVE_NOW control file
  in control/ppm-v1, 31101 RUNNING (3:06:50). C1 continues. Plan = GO (a') at local-17850 verified (~16:30).
- 14:43 PT pose5 CONFIRMED NOT PAUSED (31101 RUNNING, step 18009; no STOP_AFTER_SAVE/USR1 ever sent). Pause mechanics: pose5 touches
  control/ppm-v1/STOP_AFTER_SAVE; trainer saves at next step boundary, "STOPPED at step N"; watchdog v3 (31170) treats it as intentional (no
  relaunch); hold chain v4 (31165) idles; roller 31105 waits. Hosts free ~3 min after PAUSE NOW. pose5 releases the previous save's chain hold
  if HF-verified (+~46 GB/host). Pause save HF upload via hfb-watch small allocations on the hosts (-c 8/-c 4, 16G) -> FT must leave room or
  pose5 pauses hfb on the 8 FT hosts (risk: pause save verify > ~140 min). Resume: pose5 relaunches ppm811 from the pause save on 16 hosts,
  same env/code/alias, re-arms companions, on "RESUME 31101" or the marker. pose5 sends per-host dcp-17850 paths (fallback only).
- 14:43 PT pose5: node-local dcp-17850 only on s1, s2, s7 (slot 10), s14 (/scratch/amartyavarma/ppm/ckpt-keep-pose5/dcp-17850) and s15 (slot 9, /scratch/amartyavarma/ppmvox/init-17850-slots/dcp-17850); other 11 hosts none (Weka copy deleted 14:09 by main). HF ppm-v1-c dcp-17850 = the full copy (sha table pose5/c/hf17850/hf_files.tsv; restage tool pose5 kit/restage_dcp_hf_v2.py ~0.85 GB/s/host). C1 (HF range reads) remains the path.
- 14:43 PT B3: FT per host = 8 tasks x 4 CPUs = 32/64 CPUs, 8 GPUs, --mem 500G, not exclusive (mono8020 numbers, 3 workers/rank) -> 32 CPUs + ~92 GiB free per host -> hfb uploads keep running on FT hosts (pose5 told).
- 14:50 PT pose5 pause/resume kit READY (dry-run tested): PAUSE NOW -> save-and-exit, watchdog "stopped deliberately", pause save COMPLETE 16/16, release older verified hold, report step/save/per-host free/HOSTS FREE (~2-3 min). RESUME 31101 / marker -> relaunch from pause save on same 16 hosts, verify resume + 2 sane steps + fetcher (~10 min).
- 14:52 PT C1: HF range reads WORK (InternalCan/ppm-v1-c@c9b22137; zip central dir + data.pkl + only 'model' records, 1.34 of 4.04 GB/rank,
  CRC32-checked; stock cast/save verbatim). Verify (1) PASSED on HF dcp-17641 ranks 0, 64 (slot slinky-10 = dir slinky-7), 127 + rep.pt:
  byte-identical to existing local-17641 (sha256, torch.equal 1522/628 tensors, key order); meta differs only in src/made. Conversion job 31176
  (slinky-18, -c 2, 24G) -> local-17850.partial; first 12 ranks' sha256 == the 10-22 monoft conversion manifests (aloe-v2pure-main-17850/
  initctl). Rename gated on 129/129 match; then stock --finalize (COMPLETE), verify_local, kit preflight --mods none. ETA conversion ~15:47,
  verified ~16:00 PT. /data +83 GB.
- 14:52 PT B3 (a') link ON prepared (dry 31175): cfg vs 24272 = 10 keys (init, 7 names/paths, save_every_min, export_steps); decp_bb_off ABSENT.
  LAUNCH: bash /data/home/amartyavarma/ppm/pose6/mono17850/launch.sh --saves none --final-export-only --link-on --hosts
  slinky-15,slinky-14,slinky-5,slinky-7,slinky-1,slinky-2,slinky-17,slinky-3 --go
  VERIFY: bash /data/home/amartyavarma/ppm/pose6/mono17850/verify_final.sh /data/home/amartyavarma/ppm/pose6/mono17850/export-357 (tested PASS on
  the 17850 FT's export-200, job 31174). HF: sbatch -w <node 4 CPUs> .../mono17850/push_hf.sh <export> -> private InternalCan/ppm-pose6-mono17850-final
  (27757's uploader: sizes + sha256 + 3 read-backs).
- 14:53 PT pose6 edit mono17850/launch.sh: host check accepts a node that is not fully idle iff 0 GPUs allocated, >= 32 CPUs + >= 512000 MB free, no DRAIN/DOWN, and EVERY job there is a pose5 hfb-onhost upload (node_ok_hfb); before, node_idle would have refused the hosts during the pause-save upload. bash -n OK.

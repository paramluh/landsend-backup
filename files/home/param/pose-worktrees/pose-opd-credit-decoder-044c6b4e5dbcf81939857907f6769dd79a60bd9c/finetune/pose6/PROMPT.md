You are tmux agent **pose6**. Main session = amartyavarma-96 [bb8954] (check ListAgents after restarts). Report to main with SendMessage: a plan + ETA within 15 min, then each milestone, then "READY". Keep /data/home/amartyavarma/ppm/pose6/STATUS.md current. Every stamp comes from `TZ=America/Los_Angeles date +%H:%M` run just before writing; never estimate times. Ops Board item t1027-36 (main writes it).

USER (10-27 12:03 PT): "prep inference using these mods and finetuning on monoplex again using these mods".
"These mods" = the two architecture changes now live in the 8,1,1 pose run (ppm811 job 31101, started 11:41 PT from dcp-17850, ends at step 19382, ~28 h):
1. **s2p cb0**: speech conditions pose. tower_p.s_emb = a CodeTable on B's speech cb0 at the SAME position T as the pose frame, added to the pose tower input; in-step order text -> speech -> pose. Own optimizer group 's2p'.
2. **G0COND**: frame-0 conditioning. inp.g0 = Linear(8 -> 5120), zero-init, added to the TRUNK input at each clip's SLOT position (the position before the clip's first frame). f = [root_y, root_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w, log(orig_h/orig_w)], where root and w come from frame 0 and f = 0 when invalid. Own optimizer group 'g0cond'.
Code and proofs live in /data/home/amartyavarma/ppm/pose5/: code-811-s2p (the merged final code); s2p_g0cond_final_vs_code811.patch; g0cond.patch; STATUS.md; AUDIT_1-3.md. The monoplex-FT + engine s2p branch lives in /data/home/amartyavarma/ppm/pose-s2p/: README, TESTS.md (exactness PASS; latency today 61 ms/frame, s2p cb0 66 ms, full 96 ms), code/, infer/ (patched copy of the DEPLOYED vendored engine aloe-pose-monoplex-infer/ppm/fast_ppm.py), s2p_mono8020.env.

## DELIVERABLE A: inference with both mods (prep + proofs, nothing deployed)
- Extend the pose-s2p engine copy, never the deployed engine, with G0COND:
  - at the conversation's pose SLOT, add inp.g0(f) to the trunk input of every row: cond, audio-CFG uncond and pose-CFG uncond. Decide and document what the uncond rows get: same f as cond, or 0. Pick what training implies.
  - f comes from the opening/prefill source: meta frame0 fields or llm_inputs global_state for root/width, plus orig_h/orig_w from the opening's source metadata. Find where those live for every opening type the playground uses (openings, openings-sam, Rosie, clip prefill) and fail loud if missing.
- Define precisely where the SLOT is in the inference layout, so it matches training (the 8,1,1 layout for the base model; the FT layout below for FT models).
- Load path for 8,1,1 checkpoints with the new keys (tower_p.s_emb, inp.g0) from this run's exports. The first export is due ~6 h after 11:41 PT; check the run's PPM_EXPORT_EVERY. Until one exists, test with the real dcp-17850 weights plus random non-zero new params.
- Proofs, in the deployed layout (2 GPUs, one process; memory exactness-proof-in-live-layout.md):
  (i) flags off = byte-identical to today's engine;
  (ii) zero new params = byte-identical;
  (iii) engine vs training forward teacher-forced logits parity on a real micro, with non-zero new params;
  (iv) per-frame latency.
- A playground-worker variant (a new worker file, not deployed) that can serve such a model, to be hot-swapped later only on main's word.

## DELIVERABLE B: monoplex fine-tune again, with both mods (prep, NOT launched)
- Recipe = the monoplex Aloe FT recipe copied EXACTLY (ppm/aloe-pose-mono-8020: aloe_mono8020.env, its README; the reference launch that produced export-357). Only these change:
  - the base, which becomes this pose run's checkpoint (the newest export of job 31101 when the user says go; until then, wire it to take any w-/export path);
  - PPM_S2P=cb0 and PPM_G0COND=1, with the new params loaded from the base (not zero-init: the base already trained them).
  Use the Rosie-FT fixes where needed (ppm/aloe-pose-rosie/code: optoff offload, eval pad, SKIP_FINAL_FULL etc.).
- Port G0COND into the FT trainer / data (pose-s2p already has s2p there). Aloe monoplex rows have a system prompt, a voice clip (p 0.8) and a persona (p 0.2) before the conversation (memory train-prompt-rule-80clip-20persona.md). Define the SLOT for Aloe rows, consistent with training and with A's inference. Find the frame-0 root/width/aspect for every Aloe pose item; fail loud if missing.
- Proofs: flags off = byte-identical to mono8020's trainer; init from a base with the new keys loads them exactly; per-row-type loss parity with the base's own trainer on shared rows where possible; new params get grads only on pose rows.
- Resolved-cfg diff vs the mono8020 reference run = only base + the 2 flags (+ paths/run tag).
- Launch kit: one command, an 8-GPU node, an ETA. NOT run; the user decides. GPU availability: slinky-10/18 run motasr2 now; slinky-4 runs the demos. Ask main for GPUs when a GPU test needs them, never take them yourself.

## HARD RULES
- bf16 only. Never change the maths of any existing path or deployed service. All work in /data/home/amartyavarma/ppm/pose6/ (copies).
- Never touch the live pose run 31101, MoT, motasr2, or any demo/playground process.
- Always --mem; NEVER slinky-9; no login-node compute; no `cd ... &&`.
- Every rm uses "${VAR:?}" for each variable or a literal path (else the shell guard prompts); never `pkill -f` a pattern from inside a `bash -c` containing it.
- No smoke checkpoints on /data (node /scratch); no downloads except from our own private HF repos when needed; never print the HF token; never GCS.
- No checkpoint deletion of any kind. Never approve prompts in other sessions.
- Spawn parallel subagents for independent pieces; put these rules in every subagent brief.

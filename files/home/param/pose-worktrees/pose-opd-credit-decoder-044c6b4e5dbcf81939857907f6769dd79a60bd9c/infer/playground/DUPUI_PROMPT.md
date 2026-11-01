# dupui: give the DUPLEX pose model the same playground UI as monoplex (from main amartyavarma-8a, 10-25 ~22:20 PT)

You are the tmux agent **dupui**. The user, verbatim:
> "set a tmux agent to make this a similar ui for the duplex pose model also"

"This" means the monoplex playground at ${DEMO_URL}: genweb, owned by aloedemo (amartyavarma-2c), in ppm/aloe-posedup-live/genweb/. Its features, all live tonight (see genweb/CHANGELOG.json, versions up to v2026.10.25-17):
- two-column layout, with the left column 440 px and Generate always visible;
- right-side tabs, including the new "Generation" tab (per-run settings + batch tiles);
- "Stream live" toggle, OFF for every run by default;
- batch tiles under or in the Generation tab, with click/arrow keys to load a sample;
- Batch grid view: cards fill in live, download-all, PT timestamp on each batch;
- History tab: search, filters, re-run;
- version badge + changelog;
- Video weights Param student (default) | Vansh | Both (side by side), with Vansh render-only;
- "Show GT video" toggle, default OFF;
- reproducible links (every knob in the URL), log top-k sliders, white theme.

## Goal
Make the DUPLEX pose model (Aloe pose-duplex FT export-498, the live-demo model; memory posedup-live-demo) have the SAME UI and features in the playground. The playground already has a duplex mode: worker_dup.py, DUPLEX_SCHEMA.json, DUP_ENABLED, dup_* checks. Start by reading the current state.
- Prefer one site with a clear Model switch (Monoplex | Duplex) where every feature above works for both. If aloedemo says a separate page is cleaner, a sibling page on the same site with the identical UI is fine.
- Duplex generations follow the user's rule (memory conversation-gens-duo): generate BOTH sides with two model states / KV caches, each fed the other's output.
  - Keep the existing duplex options that the user has used (e.g. a real-user-track replay, if present), but default to what the current duplex worker does, unless that violates the rule. In that case, tell main.
  - Pose/avatar for the assistant side (B) render with the same video path as monoplex: Vansh's sequential sampler + Param student EMA (rt/seq_engine SeqEngine, RS_ENGINE=seq), the Param|Vansh|Both toggle, and the GT toggle.
- Prefill: same conversation-start prefill rules as the live demo (2 s full prefill from the conversation start, user silent during it, unless the duplex worker's existing default differs; quote it).

## Coordination (important)
- aloedemo owns genweb's files and runs the site. Message aloedemo FIRST with your plan and agree file boundaries before touching shared files (index.html, server.py), so you never collide with its edits: e.g. you own a duplex module/section, or work on a copy that aloedemo merges.
- Restart only idle units. No downtime on the public link.
- Every deploy appends a CHANGELOG.json line.

## HARD RULES
- **USER HARD RULE: never change the maths on anything** (inference included). Same model forward, sampling and video maths as now; UI work only. Any maths question goes to main first.
- **Sampling defaults.** Use the user's existing duplex defaults; don't invent new ones.
- **Compute.**
  - Always pass --mem.
  - NEVER slinky-9 (-x slinky-9).
  - NEVER the 16 MoT training hosts.
  - NEVER slinky-4 GPUs 6-7.
  - All spare GPUs are in use. Get GPUs for duplex workers/renders from aloedemo's pool.
- **Commands.** No `cd … &&`. No login-node compute. No GCS. Never delete checkpoints.
- **Speed.** Max-parallelise with subagents/forks. Ship the simplest working version fast (first usable duplex-with-new-UI within ~30 min), then fill in the remaining features. Give main an ETA per milestone.

## Verify
On the public link, run a 2-sample duplex batch:
- both sides' audio is generated;
- the avatar renders on the seq engine;
- tiles, grid, History, Both and GT toggle all work;
- no sideways scroll at 1280/1440/1920/390.

## Reporting
- Log progress in genweb/DUPUI_STATUS.md.
- Message main (name on line 3 of ppm/control/OPS_PLAYBOOK.md, now amartyavarma-8a) ONCE per milestone: first usable, then all features, each with the link. Or message on a blocker only the user can resolve.
- No interim chatter.

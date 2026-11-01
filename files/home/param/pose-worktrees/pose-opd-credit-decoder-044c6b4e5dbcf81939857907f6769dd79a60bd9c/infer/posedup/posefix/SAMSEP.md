# SAMSEP: duplex prefill codes vs training (posefix, 10-26 ~22:15 UTC; main's order; nothing deployed)

## 1. Coverage: 0 / 12 openings have SAM-isolated codes
- Duplex export-498 trains on the s2h-m1 HANDOVER rows (ALOE/s2handover-none/<uuid>/segNN/cNN; 8,107 conversations). B is the per-lane
  SAM-ISOLATED Mimi stream from data/aloe-sam-sep (HF stage2-sam-sep-v1-mimi; 12,681 handover sessions, all keys UUIDs). Sources:
  run_poseduplex.env (ALOE_SAMSEP_STORES = qwen-1ch-data/s2h-m1-samsep/stores) and aloe-ft-32-samsep/PROMPT.md ("replace ONLY the
  channel-B Mimi codes with the per-lane ISOLATED stream").
- Every opening in genweb/openings is a STUDIO recording: Aloe pose v2pure app, 40-hex recording ids, `kind: studio`.
  - Person 01: 01_ab63d745, cf6131fb57180b6c59feb66108932934989f9586 speaker_1, 0.32-12.24 s. **Not in sam-sep.**
  - The same holds for 00, 03, 08, 09, 10, 12, 13, 14, 15, 17, 19: no sam-sep row matches any recording id
    (posefix/samsep/find.py: 0 hits; sam-sep has 0 40-hex keys).
- So the duplex model never trained on any studio recording. Its B speech is handover SAM-isolated audio. The actors' openings are out of
  its training domain whatever code type they carry.

## 2. Options (nothing invented)
| option | what | status |
|---|---|---|
| (a) keep REAL codes | today | gap measured on handover: real raw-lane vs SAM-isolated codes differ on cb0 22.5 %, cb1 35 %, cb2 58 %, cb3 64 %, deeper 71-89 %, all 79 % (data/aloe-sam-sep-scripts/vc_out/report.json) |
| (b) NO audio prefill for duplex | pose + words forced, B's codes sampled for the prefill frames | available TODAY via the existing start param prefill_audio=0; or the staged env switch below |
| (c) SAM-separate the openings ourselves | upstream SAM separation + the same Mimi recipe on the opening audio | separation pipeline NOT on disk (aloe-sam-sep-scripts has only download / verify code), needs upstream code |
| (d) another SAM window of the same actor | | none: no studio actor appears in sam-sep; no identity link between studio actors and handover speakers was found |
- The same mismatch exists in the SYSTEM-PROMPT voice clip: live duplex uses gt/mimi_local REAL codes (server preset; worker_dup line ~321),
  while training draws the voice clip from another chunk of the same speaker in the samsep stores, i.e. SAM-isolated codes. Options a, b
  and c apply there too (b = no voice clip, which is off-recipe: training has voice_clip p 0.8).

## 3. Staged (NOT deployed): stage-samsep/server.py
- Base = live server.py md5 34e7f92461caeb3db1542281be60310e; staged copy md5 a1db3a779b15ab8b73febe871a6ad516 (+17 lines in prefill_inputs).
- Env PPMLIVE_DUPLEX_PREFILL_CODES:
  - `real` (default): today, byte-identical;
  - `sam`: genweb/openings_sam/<pid>/mimi_local.safetensors when present and shape-equal to the opening's codes, else no audio prefill;
  - `none`: no audio prefill.
  This server only serves the duplex model; monoplex (genweb/worker.py) is untouched and keeps the real codes.
- Check (posefix/samsep/test_switch.py, 3 actors, live vs staged prefill_inputs):
  - `real`: pose, lane, codes and info identical;
  - `none` / `sam` (no openings_sam files exist): pose and lane identical, codes absent, one note added.
- genweb/openings_sam/ was NOT created: there are no SAM-isolated codes for any opening window, so nothing to stage.
- worker_dup.py carries its own copy of the actor prefill (line ~185, the openings' mimi_local) and the real voice clip. The same switch
  would have to be ported there (aloedemo's file); not staged.

## 4. Channel A at the start: training vs live
- TRAINING (data_aloe_posedup.Stores.conv -> rowcache_vox.row_audio; VOX_A_GAIN=raw):
  - A = Voxtral of the PARTNER lane's ORIGINAL raw recording (WLT FLAC at 0 dB, the a_entry 'gcs' kind), from conversation frame 0
    (chunk-0 rows only).
  - It is not SAM-separated and not silence: in the first 2 s it carries whatever that lane has (partner speech, room tone, residual bleed).
- LIVE: A = DIGITAL ZERO for the prefill frames (engine a_mute = 1920 x max(pose / code / lane prefill frames); main's 10-25 instruction),
  then the mic.
- Mismatch: training never shows digital zero on A at the start; it shows a real lane (at minimum room tone).
  - The engine already holds a quiet-frames bank (the audio-CFG uncond row's 'quiet_frames' A table, rms 0.035) that could stand in for
    the zeros.
  - That would change inference maths: user OK needed.

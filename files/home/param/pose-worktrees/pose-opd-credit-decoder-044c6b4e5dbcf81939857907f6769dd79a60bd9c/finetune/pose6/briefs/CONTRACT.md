# pose6 inference call contract (proposed by A3 10-27; A1/A2 confirm or state differences to pose6)
A2 (pose6/infer/g0src.py):
- g0_from_opening(opening_dir: str, kind: str) -> np.float32[8]; raises on any missing field. Exports KINDS = ("openings", "openings-sam", "clip").
  - "openings": genweb/openings/<p>
  - "openings-sam": genweb/openings-sam/<p>
  - "clip": the person's avatar package dir holding gt/pose_tokens.safetensors (demo-data/prompts/<p>, or genweb/people_extra/<p> for Rosie)
- Source = the frame 0 the run's pose is decoded/rendered against (deployed worker's f0gs / sgs logic):
  opening with frame0 fields + pose prefill -> "openings"; pose-only -> openings-sam/openings, else "clip";
  no-avatar (Rosie) with src start -> "openings"; anything else -> "clip".
A1 (pose6/infer/ppm/fast_ppm.py):
- eng.start([dict(x, a_mode="empty", g0f=f)], seed, params, f0=0): g0f = per-instance dict key (np.float32[8]).
- G0COND model detected via binfo.get("g0cond") or hasattr(m.inp, "g0"); s2p via binfo["s2p"] == "cb0".

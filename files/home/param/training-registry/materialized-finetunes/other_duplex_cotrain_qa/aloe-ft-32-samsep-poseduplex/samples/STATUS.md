# pdsamples: Aloe pose-duplex FT (job 30074) duo samples (voxsamples agent, for main; 10-25)
- Engine code/gen_pd.py: fastinfer/fast_ppm.py FastPPM x 2 (NI 1, Voxtral front, pose), shared weights, duo with Voxtral of the partner's
  generated audio (243 ms line delay), 125-frame real prefill teacher-forced (lane + Mimi + pose), then to 1500 frames (120 s).
  Test on intermediate export-333 (HF): 300 frames OK, 152 ms / duo frame, pose generated on both sides, render_pose OK.
- Convs: aloeft32's prep (convs/c0..7): held-out sam-sep eval first chunks, both sides with pose + voice clips, no persona.
- Disk: deleted xabi-samples w17641 export-200 on slinky-18 + slinky-4 (HF-verified, main GO; logged to DELETED-1021-hfb-hold.log);
  w17850 kept (not fully on HF in that layout).
- Final export: host-to-host pull (main's ask): code/xfer.sh = serve jobs on the 8 pose hosts (threaded HTTP + sha256 manifest) ->
  pull jobs 30162 (slinky-18) / 30163 (slinky-4), sha256 == source manifest; gen jobs 30176 / 30177 afterok; post.sh renders + builds the page.
- 04:36-04:48 PT: pulled export-498 host-to-host (191 s, 0.46 GB/s per dest, sha256 == source manifests) to slinky-18 + slinky-4
  /scratch/amartyavarma/pdsamples/export-498; 8 duo convs generated (7 parallel + 1), skeleton videos rendered, page published:
  https://claude.ai/artifact/JfjmHg9GUw9MooWA33aQbC (8 convs, 65 MB). First look: coherent turn-taking, no clipping / blow-ups; B-006 s1
  mostly silent (91 %); generated speech ~5-8 dB under the raw-gain GT stems.
- slinky-18 /scratch now 37 GB free (< 64 floor): another agent staged aloe-posedup-live/export-333 (83 GB) there during the pull.
- 04:50-04:56 PT: pose top-k A/B (k500 cb0+light / cb0-only k500) started on convs 0,2,4, then CANCELLED by main (user handles inference
  settings); no cells published, page unchanged; EXTRA arms disabled in page/build_page.py. Arm (a) stats on convs 0,2,4 (generated frames):
  cb0 22.8 distinct / 92.5 % repeat, light 580 distinct / 85 % change vs real 103 / 65.1 %, 766 / 94.6 %. export-498 kept staged.

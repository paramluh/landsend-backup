# PAGE status: ppm-aloe-samsep-17850 live curves (ALOEFT32 page agent)

Updated 10-23 22:20 PT.

## URL
- https://movers-inspired-vocabulary-thehun.trycloudflare.com/aloe_ft_compare.html
- This is the existing klweb tunnel (klweb-ppm, ppm_serve.sbatch). If that serve job restarts, the host changes: read setup-logs/klweb/TUNNEL_URL and append /aloe_ft_compare.html.

## What changed
- Builder: setup-logs/klweb/build_aloe_ft_compare.py, extended in place. Backups: build_aloe_ft_compare.py.bak-pre-samsep-1023 and aloe_ft_compare.html.bak-pre-samsep-1023.
- The existing chart (one train normalised-word-CE curve per Aloe FT, 15-step moving average) is unchanged. Two series are added to it:
  - the live run, ink colour, thickest line, star;
  - the reference 26467 (Aloe + zoom co-train, export-253), cyan, labelled "ref".
- New section "Live run vs reference" (normalised word CE only):
  - a train chart (samsep vs 26467);
  - one eval chart per eval_*/word_ce_norm stream the live run logs. No stream names are hardcoded. Before the first eval, the reference's 4 streams are shown as empty slots.
  - Each chart has a line "At step S: samsep X vs ref Y, delta".
  - Reference-only streams (eval_zoom etc.) are listed under the charts once the run logs evals.
- Star on the live run in every chart (at the line end, or a "no points yet" note) and in the legend.
- Data:
  - The live run is read from ppm/aloe-ft-32-samsep/runs/ppm-aloe-samsep-17850-j*/metrics.jsonl. All jobs are stitched by step: per stream, a newer job replaces the older jobs' points from its own first step on.
  - The total step count comes from the newest logs/*-<JOB>.log "step N/M".
  - The job state comes from squeue, falling back to sacct.
  - The reference is read from ppm/aloe-zoom-cotrain/runs/ppm-aloe-zoom-cotrain-j26467/metrics.jsonl (train_speech, eval_speech, eval_asstB_speech, eval_userB_speech, eval_zoom).

## Comparator for the second eval stream (reference's original eval set)
- Matching rules, in order:
  1. A live stream with the same name as a reference stream is compared with that stream.
  2. Otherwise, page/series.json ref_equiv is used, e.g. {"ref_equiv": {"eval_orig_speech": "eval_speech"}}. The builder re-reads this file every build.
  3. Otherwise, a name containing "orig" or "ref" is compared with the reference's eval_speech.
- Once recipe/STATUS.md names the key, set ref_equiv explicitly if the name does not match rule 3.
- Optional "stream_labels" in series.json adds a caption per stream.

## Refresh
- No refresh was running: the old 12 h --wrap loop (job 24217) timed out on 10-19, so the page was static since then.
- New: setup-logs/klweb/aloe_ft_compare_refresh.sbatch, job 28226 (klweb-aloe-ft-compare-refresh, slinky-18, 1 CPU, no GPU, 7 days). It rebuilds every 60 s and logs to setup-logs/klweb/aloeftcompare.log.
- New j<JOB> dirs are picked up automatically; nothing needs resubmitting on resume.

## Verification (10-23 22:15 PT)
- Zero-data build OK:
  - status line "not launched yet; slot ready";
  - live-run slots present in all charts;
  - 10 legend rows (8 old + ref + samsep).
- Served through the tunnel: HTTP 200, same bytes as on disk.
- Headless Chromium (desktop + 390 px phone): no console errors, no horizontal scroll. Screenshots in page/check/shot-live-*.png.
- Synthetic resumed run (2 jobs, resume at 42, partial last line, extra eval stream "eval_origref_speech"):
  - stitched correctly;
  - the extra stream is auto-shown against the ref's eval_speech.
  - Files: page/check/fake_resumed_run.html and shot-fake-*.png (test only, not served).
- Pending: re-verify with real points after launch.

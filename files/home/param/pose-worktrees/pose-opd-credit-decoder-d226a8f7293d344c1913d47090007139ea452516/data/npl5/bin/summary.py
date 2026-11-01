"""npl5 records -> W/SUMMARY.md (re-run any time: python3 -I summary.py). Reads LISTING.json, PLACEMENT.json, state/, scan/*.json, TIMES.tsv,
plans/PLANS.json + GLOBAL_KEEP.json, dryrun/RESULT.json + GUARD*.json, launch/PLACEMENT_PHYS.json."""
import glob
import json
import os
import time

W = "/data/home/amartyavarma/ppm/pose5/npl5"
SL = "slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-7 slinky-8 slinky-9 slinky-11 slinky-12 slinky-13 slinky-14 slinky-16 slinky-17 slinky-19".split()
j = lambda p: json.load(open(p)) if os.path.exists(p) else None  # noqa: E731
L, P = j(f"{W}/LISTING.json"), j(f"{W}/PLACEMENT.json")
o = [f"# npl5 (delivery 5) staged ingest: summary ({time.strftime('%m-%d %H:%M', time.localtime())} local / see TIMES.tsv for PT)\n"]
o.append(f"- source {L['src_arg']}: {L['kept_parquet']:,} parquet + {L['kept_files'] - L['kept_parquet']:,} release.json, {L['kept_bytes'] / 1e9:.1f} GB, "
         f"{L['kept_releases']:,} releases of {L['kept_jobs']:,} jobs; 0 releases already used (plan 1 + npl2 + npl3 = {L['releases_used_before']:,})")
o.append(f"- placement (src_place.py, BALANCE=tail, PRIOR_TAGS=npl2,npl3, unconstrained free): {P['total_GB']} GB, est {P['est_new_rows']:,} rows "
         f"(~{P['est_new_steps_at_64_rows_per_host_step']} steps); per plan key GB " + ", ".join("%s:%s" % (h[7:], v["GB"]) for h, v in P["hosts"].items()))
sc = {h: j(f"{W}/scan/{h}.json") for h in SL}
done = [h for h in SL if os.path.exists(f"{W}/state/scan-{h}.ok")]
o.append(f"- scans done {len(done)}/16: {sum(sc[h]['segments'] for h in done):,} segments, {round(sum(sc[h]['frames'] for h in done) / 12.5 / 3600):,} h"
         + (f"; pending {', '.join(h for h in SL if h not in done)}" if len(done) < 16 else ""))
ow = {h: j(f"{W}/scan/origwh/{h}.json") for h in SL if os.path.exists(f"{W}/state/origwh-{h}.ok")}
o.append(f"- orig_h/orig_w sidecars (user change #2; scan/origwh/<slot>.parquet: file,row,segment_id,clip_id,orig_h,orig_w) {len(ow)}/16: "
         f"{sum(x['segments'] for x in ow.values()):,} segments, null orig_h {sum(x['null_orig_h'] for x in ow.values())}"
         + (f"; MISSED {[h for h in SL if os.path.exists(f'{W}/state/origwh-{h}.missed')]}" if any(os.path.exists(f"{W}/state/origwh-{h}.missed") for h in SL) else ""))
G, PL = j(f"{W}/plans/GLOBAL_KEEP.json"), j(f"{W}/plans/PLANS.json")
if G and PL:
    o.append(f"- global dedup (plan 1 + npl2 + npl3 scans, PPM TikTok, eval videos, SAM): scanned {G['scanned']:,}, kept {G['kept']:,}; "
             + ", ".join(f"{k} {v:,}" for k, v in G.items() if k.startswith("drop_") and not k.startswith("drop_frames")))
    o.append(f"- plans: {PL['total_rows']:,} rows, {PL['total_h']:,} h over 16 slots; src_root {PL['src_root']}; per key rows "
             + ", ".join("%s:%s" % (h[7:], v["n_rows"]) for h, v in sorted(PL["slots"].items(), key=lambda x: int(x[0][7:]))))
R = j(f"{W}/dryrun/RESULT.json")
if R:
    o.append(f"- dry run: ends {R['ends']} -> PPM_MAX_STEPS={R['max_steps']} (+{R['steps_after_resume']} steps after dcp-17850); rows/plan {R['rows_per_plan']}, "
             f"h/plan {R['hours_per_plan']}; at 17850 every rank is in plan {R['resume_in_plan']}, npl5 starts at steps {R['npl5_first_step_min_max']}; "
             f"unused rows {R['rows_unused_all_hosts']:,}; flags {R['flags']}")
for g in sorted(glob.glob(f"{W}/dryrun/GUARD*.json")):
    x = j(g); o.append(f"- {os.path.basename(g)}: {x['guard']}, append_at {x['pose_append_at']}, refusals {x.get('refusals')}, trainer {x['train_py']}")
ck = sorted(glob.glob(f"{W}/scan/checks/CHECK-*.json"))
if ck:
    o.append(f"- fresh_pose check (raw parquet == built rows / micros): {len(ck)}/16 slots OK")
LP = j(f"{W}/launch/PLACEMENT_PHYS.json")
if LP:
    o.append("- launch-day parquet share per physical host GB: " + ", ".join("%s:%s" % (p[7:], v["GB"]) for p, v in LP["shares"]["npl5"].items()))
o.append("\n## TIMES.tsv\n```\n" + (open(f"{W}/TIMES.tsv").read() if os.path.exists(f"{W}/TIMES.tsv") else "") + "```")
open(f"{W}/SUMMARY.md", "w").write("\n".join(o) + "\n")
print("\n".join(o))

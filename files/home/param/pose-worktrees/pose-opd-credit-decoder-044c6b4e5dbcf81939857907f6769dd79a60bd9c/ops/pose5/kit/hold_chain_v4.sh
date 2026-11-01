#!/bin/bash
# pose5 CHAINED HOLD (main 06:33 PT: PPM_KEEP_CKPT=1 approved ONLY with this). CPU-only loop on a spare (hold_chain.sbatch); acts on the 16 hosts
# through tiny `srun --jobid=<ppm811 job> --overlap -c1 --mem=1G --cpu-bind=none` steps. Every POLL s:
#  HOLD    every save dcp-N (N > PIN_STEP) COMPLETE on all 16 hosts under CKPT and not yet held -> hard-link it on every host into
#          /scratch/amartyavarma/ppm/ckpt-keep-pose5/hold/dcp-N (cp -al into .tmp, file count + COMPLETE inode checked, then rename; 0 B while the
#          CKPT original exists). The trainer (keep 1) prunes N right after N+1 is COMPLETE (~1 h later; earlier only if space_check is short).
#  RELEASE (v2 rule, main 10-27 ~14:05, after the hfb hold deadlock on dcp-17956): held dcp-N as soon as N itself is HF-VERIFIED 16/16
#          (setup-logs/hfb/done/dcp-N-final == "ok" AND a PASS verify record per physical host) AND a later save M > N is COMPLETE on all 16
#          hosts in CKPT (or, as before, a later save is HF-verified). Invariant: an HF-verified save always exists + the newest complete
#          node-local save. (v1 waited for a LATER save's verify; hfb refuses to hold that later save while N is still held on a tight host
#          -> deadlock.) The release is a direct srun `rm -rf --one-file-system <literal hold/dcp-N path>` on the 16 hosts (no rm inside
#          bash -c); never CKPT, never anything else. No verify -> never released.
#  v2 also: step-safe disk-trend alert (only after the hold cycle is established); STOP_HOLD_CHAIN in the state dir = clean exit between polls.
#  v3: per-save-cycle trend. v4 (audit 4): alert on the trainer's "[save] SKIPPED" (hourly); never release N while a MISMATCH hold dir exists
#      for N (R4-5); a watchdog-BAD save never counts as the later COMPLETE save (R4-6).
#  EXIT    when the run is FINISHED (watchdog state FINISHED or the log's "done <TARGET> steps") and the final save is HF-verified: release all
#          holds older than the final save (the final save stays in CKPT, not held), write state/CHAIN_DONE.
# Timeline (keep 1, hourly saves, upload+verify ~15-20 min): t0 save N COMPLETE -> <=1 poll later HOLD N -> ~t0+20m N verified (hfb; N's own
#  hfb hold released) -> t0+60m save N+1 COMPLETE, trainer prunes CKPT/dcp-N (data stays in the hold) -> ~t0+80m N+1 verified -> RELEASE N.
#  Node-local data never drops below {the newest HF-verified save} U {every newer save}; peak = N held + N+1 + 53 GB write margin only if N+1's
#  verify is late (> ~60 min) when N+2 starts.
# usage: J=<ppm811 job> [HOSTF=..] [PIN_STEP=17850] [TARGET=<max steps>] [DRY=1] [ONESHOT=1] bash hold_chain.sh
set -u
HOSTF=${HOSTF:-/data/home/amartyavarma/ppm/pose5/nodes16_s7_s15.txt}; HOSTS=$(cat $HOSTF); NH=$(echo $HOSTS | tr , '\n' | grep -c .)
CTL=/data/home/amartyavarma/ppm/control/ppm-v1; JOBF=${JOBF:-$CTL/CURRENT_JOB_811}
HFB=${HC_HFB:-/data/home/amartyavarma/setup-logs/hfb}; CK=${HC_CK:-/scratch/amartyavarma/ppm/ckpt/ppm-v1}; HD=${HC_HD:-/scratch/amartyavarma/ppm/ckpt-keep-pose5/hold}
# HC_* / HC_TEST_ROOT: offline tests only (fake hosts = dirs; __HOST__ in CK/HD is replaced per host; no srun, no squeue, no roll/export/hfb checks)
ST=${ST:-/data/home/amartyavarma/ppm/pose5/kit/state/hold_chain}; mkdir -p $ST
PIN_STEP=${PIN_STEP:-17850}; TARGET=${TARGET:-0}; POLL=${POLL:-60}; DRY=${DRY:-0}; ONESHOT=${ONESHOT:-0}
AHEAD_ALERT=${AHEAD_ALERT:-20}; MD=${MD:-/data/home/amartyavarma/ppm/control/overnight_monitor_811f.md}
LOG=$ST/hold_chain.log
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
note() { echo "[$(pt)] $*" | tee -a $LOG; }
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')
on_hosts() {   # <job> <bash script> -> stdout lines of all hosts (empty if the job is not running)
  if [ -n "${HC_TEST_ROOT:-}" ]; then local h; for h in $(echo $HOSTS | tr , ' '); do s_=${2//__HOST__/$h}; bash -c "hostname() { echo $h; }; $s_" 2>/dev/null; done; return; fi
  timeout 300 env $UNSET srun --jobid="$1" --overlap -N"$NH" -n"$NH" --ntasks-per-node=1 -w "$HOSTS" -c1 --mem=1G --cpu-bind=none \
    nice -n 10 bash -c "$2" 2>/dev/null
}
rm_on_hosts() {   # <ONE literal path> -> direct srun rm on every host (no bash -c); production: only hold/dcp-<digits>[.tmp]
  local p=${1:?} h q
  if [ -n "${HC_TEST_ROOT:-}" ]; then for h in $(echo $HOSTS | tr , ' '); do q=${p//__HOST__/$h}; rm -rf --one-file-system -- "${q:?}"; done; return 0; fi
  [[ "$p" =~ ^/scratch/amartyavarma/ppm/ckpt-keep-pose5/hold/dcp-[0-9]+(\.tmp)?$ ]] || { note "REFUSED rm of '$p' (not a hold dir)"; return 1; }
  timeout 300 env $UNSET srun --jobid="$J" --overlap -N"$NH" -n"$NH" --ntasks-per-node=1 -w "$HOSTS" -c1 --mem=1G --cpu-bind=none \
    nice -n 10 rm -rf --one-file-system -- "${p:?}" 2>/dev/null
}
verified() {   # <step> -> 0 if HF-verified on all 16 physical hosts
  local s=$1 h f
  [ "$(cat $HFB/done/dcp-$s-final 2>/dev/null)" = ok ] || return 1
  for h in $(echo $HOSTS | tr , ' '); do
    f=$(ls $HFB/verify/InternalCan__ppm-v1-*__dcp-${s}__${h}.json $HFB/verify/InternalCan__ppm-v1-*__dcp-${s}-j*__${h}.json 2>/dev/null | head -1)
    [ -n "$f" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$f')).get('PASS') is True else 1)" || return 1
  done
}
alert() {   # <key> <text>: once per key per hour -> ALERTS.log + the overnight monitor (audit 1 #6)
  local m=$ST/alert-$1; [ -f $m ] && [ $(( $(date +%s) - $(stat -c %Y $m) )) -lt 3600 ] && return 0
  note "ALERT $2"; [ "$DRY" = 1 ] && return 0
  [ -n "${HC_TEST_ROOT:-}" ] && { echo "TESTALERT $2" >> $ST/test_alerts; touch $m; return 0; }   # tests never write the real alert logs
  echo "- $(pt): [pose5 ALERT] $2" | tee -a $MD >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; touch $m
}
note "armed: hosts $NH, pin step $PIN_STEP, target $TARGET, poll ${POLL}s, dry $DRY"
while :; do
  if [ -f "$ST/STOP_HOLD_CHAIN" ]; then mv -f "$ST/STOP_HOLD_CHAIN" "$ST/STOP_HOLD_CHAIN.done-$(date +%s)"; note "STOP_HOLD_CHAIN: clean exit between polls (no hold op in flight)"; exit 0; fi
  J=${J_FIXED:-$(cat $JOBF 2>/dev/null)}
  if [ -n "$J" ] && { [ -n "${HC_TEST_ROOT:-}" ] || squeue -h -j "$J" -t R 2>/dev/null | grep -q .; }; then
    OUT=$(on_hosts "$J" "h=\$(hostname -s); for d in $CK/dcp-*; do n=\${d##*-}; [[ \$n =~ ^[0-9]+\$ ]] && [ -f \$d/COMPLETE ] && echo \"L \$h \$n\"; done; \
for d in $HD/dcp-*; do n=\${d##*-}; [[ \$n =~ ^[0-9]+\$ ]] && [ -f \$d/COMPLETE ] || continue; \
if [ -f $CK/dcp-\$n/COMPLETE ] && [ \$(stat -c %i $CK/dcp-\$n/COMPLETE) != \$(stat -c %i \$d/COMPLETE) ]; then echo \"X \$h \$n\"; else echo \"H \$h \$n\"; fi; done; echo \"E \$h\"")
    if [ "$(echo "$OUT" | grep -c '^E ')" = "$NH" ]; then
      LIVE=$(echo "$OUT" | awk '$1=="L"{print $3}' | sort -n | uniq -c | awk -v n=$NH -v p=$PIN_STEP '$1==n && $2>p{print $2}')
      for s in $LIVE; do grep -qE "^[0-9]+ $s\$" "$ST/save_events.tsv" 2>/dev/null || echo "$(date +%s) $s" >> "$ST/save_events.tsv"; done   # v3: save COMPLETE events (first seen)
      # audit 2 R2-3: HELD = held on ALL hosts (count filter; one per line, exact-line membership); HELDANY = held on >= 1 host (partial
      # holds are retried, alerted, timestamped and released like full ones)
      HELD=$(echo "$OUT" | awk '$1=="H"{print $3}' | sort -n | uniq -c | awk -v n=$NH '$1==n{print $2}')
      HELDANY=$(echo "$OUT" | awk '$1=="H"{print $3}' | sort -n | uniq)
      # a hold dir whose COMPLETE is not the CKPT save's (same step, other inode) does NOT count as held (X): alerted, never touched
      echo "$OUT" | awk '$1=="X"{print $3, $2}' | sort | awk '{a[$1] = a[$1] (a[$1] ? "," : "") $2} END {for (s in a) print s, a[s]}' | while read s hs; do
        alert mismatch-$s "hold dir dcp-$s on $hs is NOT this save's hard links (COMPLETE inode differs from CKPT's): left untouched, KEEP=1 net down there: main decides"; done
      for s in $HELDANY; do [ -f $ST/held-$s ] || [ -f $ST/released-$s ] || echo "$(date +%s)" > $ST/held-$s; done   # first hold seen (any host)
      for s in $LIVE; do                                                    # HOLD
        if printf '%s\n' $HELD | grep -qx "$s"; then rm -f "${ST:?}/unheld-${s:?}"; continue; fi
        [ -f "$ST/released-$s" ] && continue                                # released once (a later save HF-verified): never re-held
        k=$(echo "$OUT" | awk -v s=$s '$1=="H" && $3==s' | wc -l)
        echo x >> $ST/unheld-$s; [ "$(wc -l < $ST/unheld-$s)" -gt 2 ] && alert unheld-$s "save dcp-$s COMPLETE on all $NH hosts but held on only $k/$NH after 2 polls (KEEP=1 safety net down$([ $k -gt 0 ] && echo ': PARTIAL hold'))"
        if [ "$DRY" = 1 ]; then note "DRY would hold dcp-$s on $((NH - k)) hosts (held on $k/$NH)"; continue; fi
        # per host: already held (same COMPLETE inode) -> OK; a hold dir that is NOT this save's hard links -> MISMATCH, left untouched (never
        # deleted here: main decides); else cp -al into .tmp (ours), verify count + inode, mv -T (never into an existing dir)
        rm_on_hosts "${HD:?}/dcp-${s:?}.tmp"                               # a stale .tmp of ours (direct srun rm, literal path)
        R=$(on_hosts "$J" "mkdir -p $HD; if [ -f $HD/dcp-$s/COMPLETE ] && [ \$(stat -c %i $CK/dcp-$s/COMPLETE) = \$(stat -c %i $HD/dcp-$s/COMPLETE) ]; then echo OK \$(hostname -s) already; \
elif [ -e $HD/dcp-$s ]; then echo MISMATCH \$(hostname -s); \
elif [ -e $HD/dcp-$s.tmp ]; then echo TMPLEFT \$(hostname -s); \
else cp -al $CK/dcp-$s $HD/dcp-$s.tmp && [ \$(ls $CK/dcp-$s | wc -l) = \$(ls $HD/dcp-$s.tmp | wc -l) ] \
&& [ \$(stat -c %i $CK/dcp-$s/COMPLETE) = \$(stat -c %i $HD/dcp-$s.tmp/COMPLETE) ] && mv -T $HD/dcp-$s.tmp $HD/dcp-$s && echo OK \$(hostname -s); fi")
        n=$(echo "$R" | grep -c '^OK '); note "HOLD dcp-$s: $n/$NH hosts$([ $n = $NH ] || echo ' (INCOMPLETE: retried next poll)')"
        mm=$(echo "$R" | awk '$1=="MISMATCH"{print $2}' | paste -sd,); [ -n "$mm" ] && alert mismatch-$s "hold dir dcp-$s on $mm is NOT this save's hard links (COMPLETE inode differs / incomplete): left untouched, KEEP=1 net down there: main decides"
        [ -f $ST/held-$s ] || echo "$(date +%s)" > $ST/held-$s
      done
      for s in $HELDANY; do                                                 # no HF verify within 2 h of the (first) hold (audit 1 #6; partial holds too)
        [ -f $ST/held-$s ] && [ $(( $(date +%s) - $(cat $ST/held-$s) )) -gt 7200 ] && ! verified $s && alert noverify-$s "held save dcp-$s has no HF verify 2 h after its hold (hfb-watch stuck?)"
      done
      XS=$(echo "$OUT" | awk '$1=="X"{print $3}' | sort -u)                 # v4 (audit 4 R4-5): steps with a MISMATCH hold dir on some host
      BADS=$(ls "${WD_ST:-/data/home/amartyavarma/ppm/control/ppm-v1/watchdog_pose5_state}" 2>/dev/null | grep -oE '^bad-[0-9]+$' | cut -d- -f2)
      for s in $HELDANY; do                                                 # RELEASE (full or partial holds)
        [[ "$s" =~ ^[0-9]+$ ]] || continue
        if printf '%s\n' $XS | grep -qx "$s"; then note "RELEASE of dcp-$s skipped: a MISMATCH hold dir exists for it on some host (left untouched for main)"; continue; fi
        why=""
        # v4 (audit 4 R4-6): a save the watchdog marked BAD never counts as "the later save COMPLETE 16/16"
        newer=$(printf '%s\n' $LIVE | grep -vxF -f <(printf '%s\n' $BADS "x") | awk -v s=$s '$1 > s' | tail -1)   # a later good save COMPLETE 16/16 in CKPT
        if [ -n "$newer" ] && verified $s; then why="dcp-$s HF-verified 16/16 + dcp-$newer COMPLETE 16/16 (v2 rule)"
        else
          later=$(for m in $(ls $HFB/done 2>/dev/null | grep -oE '^dcp-[0-9]+-final$' | grep -oE '[0-9]+'); do [ "$m" -gt "$s" ] && echo $m; done | sort -n)
          for m in $later; do verified $m && { why="later save dcp-$m HF-verified 16/16"; break; }; done
        fi
        [ -z "$why" ] && continue
        if [ "$DRY" = 1 ]; then note "DRY would release dcp-$s ($why)"; continue; fi
        rm_on_hosts "${HD:?}/dcp-${s:?}"                                     # direct srun rm, literal path (no rm inside bash -c)
        R=$(on_hosts "$J" "[ -e $HD/dcp-$s ] || echo OK \$(hostname -s)")
        n=$(echo "$R" | grep -c '^OK '); note "RELEASE dcp-$s ($why): $n/$NH hosts"
        [ $n = $NH ] && mv -f $ST/held-$s $ST/released-$s 2>/dev/null && { touch "$ST/released-$s"; [ -f "$ST/first_release_at" ] || date +%s > "$ST/first_release_at"; }   # v2: release time (mv keeps the hold's mtime)
      done
    else note "host scan incomplete ($(echo "$OUT" | grep -c '^E ')/$NH); retry"; fi
  fi
  # npl5 roll watch (main 10-27): B's daemons write npl5/roll/<host>.json (ahead_steps = covered - live); collect them for the live page and
  # alert (once per host per hour) when any host is < AHEAD_ALERT (20) steps ahead; B's own BEHIND marker fires at 30.
  [ -n "${HC_TEST_ROOT:-}" ] && { [ "$ONESHOT" = 1 ] && exit 0; sleep $POLL; continue; }   # tests: hold / release only
  # audit 3 R3-4 / main 10-27: Weka-served hosts' /scratch free for the trend alert. B's roll/<host>.json carries scratch_free_GB (= /scratch)
  # + dest_free_GB (= the resolved dest, /data) from fetcher v2.2; an older file (no dest_free_GB) reports /data as scratch_free_GB -> fall
  # back to a df of /scratch on that host (tiny overlap step on the running ppm811 job)
  WKDF=""
  if [ -n "$J" ] && squeue -h -j "$J" -t R 2>/dev/null | grep -q .; then
    for h in $(python3 -I /data/home/amartyavarma/ppm/pose5/kit/gate_rules.py wekahosts); do
      grep -q '"dest_free_GB"' /data/home/amartyavarma/ppm/pose5/npl5/roll/$h.json 2>/dev/null && continue
      g=$(timeout 120 env $UNSET srun --jobid="$J" --overlap -N1 -n1 -w "$h" -c1 --mem=1G --cpu-bind=none df -B1 --output=avail /scratch 2>/dev/null | tail -1 | tr -dc 0-9)
      [ -n "$g" ] && WKDF="$WKDF $h=$g"
    done
  fi
  python3 - "$AHEAD_ALERT" "$MD" "$ST" "$WKDF" <<'EOF' 2>>$LOG
import glob, json, os, sys, time
os.environ["TZ"] = "America/Los_Angeles"; time.tzset()
lim, md, st = int(sys.argv[1]), sys.argv[2], sys.argv[3]
wkdf = {k: int(v) / 1e9 for k, _, v in (x.partition("=") for x in sys.argv[4].split()) if v.isdigit()}
sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/kit"); import gate_rules as GR
R = os.environ.get("HC_ROLL_DIR", "/data/home/amartyavarma/ppm/pose5/npl5/roll"); rows = {}   # HC_ROLL_DIR / HC_ALERTS: offline tests only
AL = os.environ.get("HC_ALERTS", "/data/home/amartyavarma/setup-logs/watchdog/ALERTS.log")
for f in glob.glob(f"{R}/slinky-*.json"):
    if f.count(".") > 1: continue
    try:
        j = json.load(open(f)); ph = j.get("phys", os.path.basename(f)[:-5])
        fr, src = j.get("scratch_free_GB"), "roll json"
        if ph in GR.WEKA_HOSTS and "dest_free_GB" not in j:       # old format: scratch_free_GB is the Weka dest (/data) -> use df of /scratch
            fr, src = wkdf.get(ph), ("df /scratch (overlap step)" if ph in wkdf else "none (no df this poll)")
        rows[ph] = {"ahead": j.get("ahead_steps"), "live": j.get("live_step"), "free_GB": fr, "free_src": src, "dest_free_GB": j.get("dest_free_GB"),
                    "weka": ph in GR.WEKA_HOSTS, "GB": j.get("GB_present"), "age_s": round(time.time() - os.path.getmtime(f))}
    except Exception:
        pass
def alert(key, text):                                   # once per key per hour
    m = f"{st}/alert-{key}"
    if os.path.exists(m) and time.time() - os.path.getmtime(m) < 3600: return
    line = f"- {time.strftime('%a %m-%d %H:%M %Z')}: [pose5 ALERT] {text}"
    open(md, "a").write(line + "\n"); open(AL, "a").write(line + "\n"); open(m, "w").write(line)
# /scratch trend per host (main 10-27, esp. slinky-13): history of B's scratch_free_GB; alert if free < floor + margin, or if the last hour's
# slope projects free to cross floor + 5 GB within 2 h. Rules = kit/gate_rules.py (floor s13 20 / 30, margin s13 3.5 / 5). Weka-served hosts:
# their /scratch (the saves), never the /data dest (the fetcher's own 350 GB floor + its NOSPACE marker guard /data).
# v2 (10-27, after 11 false positives at 12:40): the first pose save (+46 GB, dcp-17850 was off-node) and the first time the chained hold keeps
# 2 saves (+46 GB) are one-time steps, not a trend. The TREND is evaluated only once the hold cycle is established -- the first release
# (= a later save HF-verified and the earlier hold dropped; st/first_release_at) -- and only on two FULL hourly windows after it, so both
# troughs are the same state (N held + N+1) = like for like. The absolute floor + margin check stays on from the start.
fra = f"{st}/first_release_at"
if not os.path.exists(fra) and glob.glob(f"{st}/released-*"):
    open(fra, "w").write(f"{time.time():.0f}\n")      # v2 started after v1 already released (mv kept the hold's mtime): count from now (later = safe)
est = float(open(fra).read().split()[0]) if os.path.exists(fra) else None
# v3 STEP-AWARE (main 10-27): one trough per SAVE CYCLE (save k COMPLETE -> save k+1 COMPLETE; events = st/save_events.tsv, first seen by the
# hold scan), cycles counted only once the hold cycle is established (they start at/after first_release_at); compare the last two COMPLETE
# cycles: same state every cycle (N+1 complete, N held until verified) = like for like, whatever the save cadence (hourly windows could hold
# 0 or 2 troughs when the cadence drifts). Alert if the last trough fell and 2 more such falls would cross floor + 5.
try:
    ev = sorted((float(l.split()[0]), int(l.split()[1])) for l in open(f"{st}/save_events.tsv") if len(l.split()) == 2)
except OSError:
    ev = []
cyc = [(t0, t1) for (t0, _), (t1, _) in zip(ev, ev[1:]) if est is not None and t0 >= est]
warn = {}
for h, r in rows.items():
    fr = r.get("free_GB")
    if fr is None or r["age_s"] > 900: continue
    hf = f"{st}/free-{h}.tsv"; open(hf, "a").write(f"{time.time():.0f}\t{fr}\n")
    pts = [tuple(map(float, l.split())) for l in open(hf) if len(l.split()) == 2][-2000:]
    tr = [min(v for t, v in pts if t0 <= t < t1) for t0, t1 in cyc if any(t0 <= t < t1 for t, _ in pts)]
    flo, mar = GR.floor(h), GR.margin(h)
    why = None
    if fr < flo + mar: why = f"/scratch free {fr:.1f} GB < floor {flo:.0f} + margin {mar}"
    elif len(tr) >= 2 and tr[-1] < tr[-2] and tr[-1] + 2 * (tr[-1] - tr[-2]) < flo + 5:
        why = (f"/scratch per-save-cycle troughs falling {tr[-2]:.1f} -> {tr[-1]:.1f} GB: projected below floor {flo:.0f} + 5 GB within 2 cycles "
               f"(now {fr:.1f} GB)")
    r["troughs"] = [round(x, 1) for x in tr[-4:]]
    if why:
        warn[h] = round(fr, 1); alert(f"disk-{h}", f"{h}: {why}")
json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "alert_below": lim, "hosts": rows, "disk_warn": warn,
           "trend": f"per save cycle; {len(cyc)} complete cycle(s) since the hold cycle was established "
                    + (time.strftime("%H:%M %Z", time.localtime(est)) if est else "(not yet: first release pending)") + "; alerts need >= 2"},
          open(f"{st}/roll_ahead.json", "w"), indent=1)
# B fetcher v2 (10-27): markers, step death, stale status files
# audit 3 R3-1 / main 10-27: NOSPACE is DEBOUNCED (the hourly save -> hold-release window pauses fetching for ~15 min by design): alert only
# if a host's NOSPACE has lasted > 30 min (first seen, tracked in st/nospace-first-<host>) OR its lead is < 50 steps. Others unchanged.
nos = set()
for f in glob.glob(f"{R}/slinky-*.*"):
    b = os.path.basename(f); h, _, mk = b.partition(".")
    if mk == "NOSPACE":
        nos.add(h); ff = f"{st}/nospace-first-{h}"
        if not os.path.exists(ff): open(ff, "w").write(f"{time.time():.0f}\n")
        age = max(0.0, time.time() - float(open(ff).read().split()[0])); a = (rows.get(h) or {}).get("ahead")
        if age > 1800 or (a is not None and a < 50):
            alert(f"roll-{h}-NOSPACE", f"npl5 fetcher NOSPACE on {h} for {age / 60:.0f} min, lead {a} steps (debounced: > 30 min or < 50 steps) "
                                      f"({open(f).read()[:160].strip()})")
    elif mk in ("BEHIND", "CRIT", "STARVED", "LOWMEM"):
        alert(f"roll-{h}-{mk}", f"npl5 fetcher {mk} on {h} ({open(f).read()[:160].strip()})")
for ff in glob.glob(f"{st}/nospace-first-*"):                 # NOSPACE cleared -> reset its clock
    if os.path.basename(ff)[len("nospace-first-"):] not in nos: os.remove(ff)
if os.path.exists(f"{R}/STEP_DIED"):
    alert("roll-stepdied", f"npl5 fetcher: a per-host daemon step DIED ({open(f'{R}/STEP_DIED').read()[:160].strip()})")
for h, r in rows.items():
    if r["age_s"] > 900: alert(f"roll-{h}-stale", f"npl5 fetcher status roll/{h}.json is {r['age_s']} s old (> 900): daemon stuck or dead?")
sp = f"{R}/SUPER.json"
if os.path.exists(sp) and time.time() - os.path.getmtime(sp) > 300:
    alert("roll-super-stale", f"npl5 roll supervisor status SUPER.json is {round(time.time() - os.path.getmtime(sp))} s old (> 300): supervisor dead?")
for h, r in rows.items():
    a = r["ahead"]
    if a is not None and a < lim and r["age_s"] < 900:
        m = f"{st}/alert-ahead-{h}"
        if not os.path.exists(m) or time.time() - os.path.getmtime(m) > 3600:
            line = f"- {time.strftime('%a %m-%d %H:%M %Z')}: [pose5 ALERT] npl5 roll fetcher on {h} only {a} steps ahead of the trainer (< {lim}; live {r['live']}); a missing file kills the job (watchdog relaunch)"
            open(md, "a").write(line + "\n"); open(AL, "a").write(line + "\n"); open(m, "w").write(line)
EOF
  L=/data/home/amartyavarma/ppm/logs/ppm-$J.log
  # export /data drop (main 10-27 = MoT's policy): kit/export_drop.py, background, one at a time (flock); live HF re-check + sha256 inside
  ( flock -n 9 || exit 0; O=$(nice -n 10 /data/home/amartyavarma/venvs/duplex/bin/python -I /data/home/amartyavarma/ppm/pose5/kit/export_drop.py $([ "$DRY" = 1 ] && echo --dry-run) 2>&1)
    echo "$O" | sed "s/^/[$(pt)] /" >> $ST/export_drop.log
    echo "$O" | grep -q 'ALERT' && alert exportdrop "export drop: $(echo "$O" | grep ALERT | head -1 | cut -c1-200)" ) 9>$ST/export_drop.lock &
  squeue -h -u amartyavarma -n hfb-watch -t R -o %i | grep -q . || alert hfbwatch "HF watcher hfb-watch is NOT RUNNING (no save of the pose run gets backed up; holds pile up)"
  # v4 (audit 4 R4-4): the trainer's "!!! [save] SKIPPED" (a host's /scratch cannot hold the next save) -> ALERT, at most hourly while new
  # skips keep coming (a skipped save also stalls the v2 release rule: no later save COMPLETE)
  SK=$(tail -c 20000000 "$L" 2>/dev/null | grep -a '\[save\] SKIPPED' | tail -1)
  if [ -n "$SK" ] && [ "$SK" != "$(cat "$ST/save_skipped_last" 2>/dev/null)" ]; then
    m_=$ST/alert-saveskipped
    if [ ! -f "$m_" ] || [ $(( $(date +%s) - $(stat -c %Y "$m_") )) -ge 3600 ]; then
      alert saveskipped "trainer SKIPPED a save (some host's /scratch too full; previous save kept): $(echo "$SK" | cut -c1-220)"
      [ "$DRY" = 1 ] || echo "$SK" > "$ST/save_skipped_last"
    fi
  fi
  G=$(tail -c 400000 "$L" 2>/dev/null | grep -a "\] step [0-9]*/" | tail -1)                 # audit 1 #7: clip-group grad norms
  for k in audio tower_p; do v=$(echo "$G" | grep -oE "train/gn_$k=[0-9.]+" | cut -d= -f2)
    [ -n "$v" ] && awk -v v=$v 'BEGIN{exit !(v > 0.8)}' && alert gn-$k "train/gn_$k = $v > 0.8 at $(echo "$G" | grep -oE 'step [0-9]+') (per-group clip 1.0 may start rescaling the group's other updates)"
  done
  if [ "$TARGET" -gt 0 ] && grep -aq "done $TARGET steps" "$L" 2>/dev/null && verified "$TARGET"; then
    note "run FINISHED at $TARGET, final save dcp-$TARGET HF-verified 16/16: exiting; holds left (pose-run files, kept; removal only on main's word if MoT needs the space): $(ls $ST/held-* 2>/dev/null | grep -oE '[0-9]+$' | paste -sd' ')"
    touch $ST/CHAIN_DONE; exit 0
  fi
  [ "$ONESHOT" = 1 ] && exit 0
  sleep $POLL
done

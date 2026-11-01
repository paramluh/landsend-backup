#!/bin/bash
# aloe-posedup-live verification, run INSIDE the demo job: srun --overlap --jobid <demo job> -n1 -c2 --cpu-bind=none bash tools/verify.sh <out> [secs]
# 1. a local test feed endpoint (tools/feed_sink.py :9199) <- FEED_URL.txt for this check only (restored after);
# 2. ONE NON-PREEMPTING test session (client 'test') through the public plain URL with a held-out 125 s wav as the mic (default preset);
# 3. ECAPA voice match of B's audio vs every preset clip (a16861 voicecheck.py); 4. the sink's feed messages vs the server's feed.jsonl.
H=/data/home/amartyavarma/ppm/aloe-posedup-live; O=$1; SECS=${2:-120}; mkdir -p $O
source $H/env_live.sh
U=$(cat $H/URL.txt); W=/data/home/amartyavarma/ppm/aloe-m1vox-live/a16861/verify-2min/a_125s.wav
echo "URL $U  $(date)"
OLD=""; [ -f $H/FEED_URL.txt ] && OLD=$(cat $H/FEED_URL.txt)
nice -n 5 $PY -I $H/tools/feed_sink.py 9199 $O/sink.jsonl > $O/sink.log 2>&1 & SP=$!
sleep 2; echo "ws://127.0.0.1:9199/feed" > $H/FEED_URL.txt
curl -s --max-time 20 $U/api/presets > $O/presets.json
PID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['default_preset'])" $O/presets.json); echo "default preset $PID"
PPMLIVE_PERSON=${PERSON:-} nice -n 5 $PY -I $H/test_client.py $U $W $O/s1 $SECS > $O/s1.log 2>&1; echo "s1 rc=$? $(grep -m1 ready $O/s1.log)"
sleep 3
if [ -n "$OLD" ]; then echo "$OLD" > $H/FEED_URL.txt; else rm -f $H/FEED_URL.txt; fi
kill $SP 2>/dev/null; wait $SP 2>/dev/null
python3 - $O <<'PY'
import json, sys, glob
O = sys.argv[1]
r = json.load(open(f"{O}/s1/result.json"))
print("session", r["session"], "frames", r["frames"], "b_s", r["b_seconds"], "p50", round(r["ms_p50"], 1), "p95", round(r["ms_p95"], 1), "max", round(r["ms_max"], 1), "rtf", r["rtf"], "lag_max", r["lag_max"])
print("pose", r["pose"])
print("words", repr(r["words"][:400]))
for f in glob.glob(f"{O}/sink.jsonl.summary-*.json"):
    print("sink", open(f).read().replace("\n", " ")[:600])
PY
SID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['session'])" $O/s1/result.json)
python3 - $O $H/sessions/$SID <<'PY'
import json, sys
O, S = sys.argv[1], sys.argv[2]
sink = [json.loads(l) for l in open(f"{O}/sink.jsonl")]
srv = [json.loads(l) for l in open(f"{S}/feed.jsonl")]
for d in sink: d.pop("_rx", None)
sid = srv[0]["session"]
sink = [d for d in sink if d.get("session") == sid]
print("feed.jsonl", len(srv), "msgs; sink got", len(sink), "; identical:", sink == srv, "; types", {t: sum(1 for d in srv if d["type"] == t) for t in ("start", "f", "end")})
print("start", srv[0]); print("end", srv[-1])
sj = json.load(open(f"{S}/session.json")); print("session.json feed", sj.get("feed"), "pose_frames", sj.get("pose_frames"), "missing", sj.get("pose_missing_frames"), "end_reason", sj.get("end_reason"))
PY
nice -n 5 /data/home/amartyavarma/venvs/voicebank/bin/python -I /data/home/amartyavarma/ppm/aloe-m1vox-live/a16861/tools/voicecheck.py $H/clips /scratch/amartyavarma/aloedemo-vc $O/voice.json $O/s1:$PID > $O/voice.log 2>&1; echo "voice rc=$?"; grep -E '"chosen"|sim_chosen|best_other|PASS' $O/voice.log | head -8

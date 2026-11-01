#!/bin/bash
# Test samdata_register.py WITHOUT touching the live page (run on slinky-18 via srun, CPU only):
#  1 --dry for S=17852 / aloev2pm17852 (patched copies -> dryrun/samdata/17852/)
#  2 the dry copies == patch.diff (placeholders filled, label 17,852) applied by `patch -F0` to copies of the live files
#  3 the dry builder runs end to end, outputs to dryrun/samdata/17852/out; JSON has p17852 (no run dir yet)
#  4 node --check on the inline <script> of the generated page (and of the live page, as a baseline)
#  5 live-mode logic on a sandbox copy (SAMDATA_REG_KLWEB): apply, no-op, refuse, --replace, injected failure -> restore, missing anchor
#  6 the live files' hashes are unchanged
set -u
T=/data/home/amartyavarma/ppm/monoft/tools/samdata_register.py
D=/data/home/amartyavarma/ppm/monoft/dryrun/samdata
K=/data/home/amartyavarma/setup-logs/klweb
P=$D/patch.diff.in   # copy of the investigation's patch.diff (the scratchpad /tmp is not visible on slinky-18)
NODE=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node
S=17852; J=aloev2pm17852
X=$D/test; rm -rf $X; mkdir -p $X
ok() { echo "PASS $*"; }; bad() { echo "FAIL $*"; FAILS=$((FAILS+1)); }; FAILS=0
LIVE0=$(sha1sum $K/build_samdata_ft.py $K/samdata_ft.tpl.html)

python3 -c "import py_compile,sys; py_compile.compile('$T', cfile='$X/reg.pyc', doraise=True)" && ok "py_compile samdata_register.py" || bad "py_compile"

# 1 dry
python3 $T $S $J --dry > $X/dry.out 2>&1; rc=$?
[ $rc -eq 0 ] && ok "dry rc 0 ($(grep -c '^[+-][^+-]' $X/dry.out) +/- lines)" || { bad "dry rc $rc"; cat $X/dry.out; }

# 2 equivalence with patch.diff
mkdir -p $X/patchref; cp $K/build_samdata_ft.py $K/samdata_ft.tpl.html $X/patchref/
sed -e "s/<S>/$S/g; s/<JOBNAME>/$J/g; s/from step $S (8,1,1/from step 17,852 (8,1,1/" $P > $X/patch-$S.diff
[ -s $X/patch-$S.diff ] && patch -s -F0 -d $X/patchref -p1 < $X/patch-$S.diff && ok "patch.diff applies -F0" || bad "patch.diff apply"
for f in build_samdata_ft.py samdata_ft.tpl.html; do cmp -s $X/patchref/$f $D/$S/$f && ok "dry $f == patch.diff result" || { bad "dry $f differs from patch.diff"; diff $X/patchref/$f $D/$S/$f | head; }; done

# 3 the dry builder end to end (W = its own dir -> reads the patched template; out dir = argv[1])
mkdir -p $D/$S/out
[ ! -e /data/home/amartyavarma/ppm/aloe-v2pure-main-$S ] && ok "run dir aloe-v2pure-main-$S absent (as expected)" || echo "NOTE run dir exists"
timeout 240 python3 $D/$S/build_samdata_ft.py $D/$S/out > $X/build.out 2>&1; rc=$?
[ $rc -eq 0 ] && ok "dry builder rc 0" || bad "dry builder rc $rc"
tail -2 $X/build.out | cut -c1-600
python3 - $D/$S/out $K/samdata_ft_live.json $S <<'EOF'
import json, sys
o, livep, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = json.load(open(f"{o}/samdata_ft_live.json")); L = json.load(open(livep)); pid = f"p{S}"
def chk(c, m): print(("PASS " if c else "FAIL ") + m)
f = (s.get("fts") or {}).get(pid)
chk(f is not None, f"json fts.{pid} present: " + json.dumps({k: f.get(k) for k in ("phase", "nojob", "init", "live", "done", "max_steps")} if f else None))
chk(not str((f or {}).get("phase", "")).startswith("builder error"), "no builder error for the missing run dir")
sc = {q["id"]: q for q in s.get("scaling") or []}
chk(pid in sc and list(sc)[-1] == pid, f"scaling has {pid} (last in SC_ORDER): " + json.dumps(sc.get(pid)))
chk("scaling_err" not in s, "no scaling_err")
chk(any(r["id"] == pid for r in s.get("lr") or []), f"lr has {pid}")
same = all(json.dumps(sc[k], sort_keys=True) == json.dumps(q, sort_keys=True) for k, q in ((q["id"], q) for q in L.get("scaling") or []))
chk(same and set(sc) == {q["id"] for q in L.get("scaling") or []} | {pid}, "other scaling points identical to the live JSON")
chk(set(s["fts"]) == set(L["fts"]) | {pid}, f"fts keys = live + {pid}: {sorted(s['fts'])}")
h = open(f"{o}/samdata_ft.html").read()
chk(f"'{pid}'" in h and "--cm:#5c940d" in h and "PPM 32B from step 17,852 (8,1,1 + delivery 3)" in h, "html carries the p17852 entries")
EOF

# 4 node --check on the inline script (patched page + live page baseline)
for pg in $D/$S/out/samdata_ft.html $K/samdata_ft.html; do
  nm=$(basename $(dirname $pg))
  python3 -c "import re,sys; h=open(sys.argv[1]).read(); s=re.findall(r'<script>(.*?)</script>', h, re.S); assert len(s)==1, len(s); open(sys.argv[2],'w').write(s[0])" $pg $X/inline-$nm.js
  $NODE --check $X/inline-$nm.js && ok "node --check inline script ($nm page, $(wc -c < $X/inline-$nm.js) bytes)" || bad "node --check $nm"
done

# 5 live-mode logic on a sandbox copy
SB=$X/sandbox; PR=$X/pristine; mkdir -p $SB $PR; cp $K/build_samdata_ft.py $K/samdata_ft.tpl.html $PR/
reset() { cp $PR/build_samdata_ft.py $PR/samdata_ft.tpl.html $SB/; rm -f $SB/*.bak-monoft-*; }
run() { SAMDATA_REG_KLWEB=$SB python3 $T "$@" > $X/sb.out 2>&1; echo $?; }
same_as() { cmp -s $SB/build_samdata_ft.py $1/build_samdata_ft.py && cmp -s $SB/samdata_ft.tpl.html $1/samdata_ft.tpl.html; }
reset
r=$(run $S $J); [ $r -eq 0 ] && same_as $D/$S && [ -f $SB/build_samdata_ft.py.bak-monoft-$S ] && cmp -s $SB/samdata_ft.tpl.html.bak-monoft-$S $PR/samdata_ft.tpl.html \
  && ok "sandbox apply: rc 0, == dry copies, backups = originals" || { bad "sandbox apply rc $r"; cat $X/sb.out; }
ls -a $SB | grep -q '\.reg-' && bad "temp files left" || ok "no temp files left"
r=$(run $S $J); [ $r -eq 0 ] && grep -q no-op $X/sb.out && same_as $D/$S && ok "second call: no-op ($(cat $X/sb.out))" || { bad "second call rc $r"; cat $X/sb.out; }
r=$(run 17900 aloev2pm17900); [ $r -eq 2 ] && same_as $D/$S && ok "different S: refused rc 2 ($(head -c 150 $X/sb.out))" || { bad "different S rc $r"; cat $X/sb.out; }
r=$(run $S aloev2pmX); [ $r -eq 2 ] && same_as $D/$S && ok "same S, other job name: refused rc 2" || { bad "same S other job rc $r"; cat $X/sb.out; }
r=$(run 17900 aloev2pm17900 --replace); mkdir -p $X/ref17900; cp $SB/build_samdata_ft.py $SB/samdata_ft.tpl.html $X/ref17900/
reset; r2=$(run 17900 aloev2pm17900)
[ $r -eq 0 ] && [ $r2 -eq 0 ] && same_as $X/ref17900 && ok "--replace 17852 -> 17900 == fresh 17900 registration" || bad "--replace rc $r / $r2"
reset; r=$(SAMDATA_REG_FAILTEST=1 run $S $J); [ $r -eq 1 ] && same_as $PR && grep -q restored $X/sb.out && ok "injected failure after write: rc 1, both files restored" || { bad "failtest rc $r"; cat $X/sb.out; }
reset; sed -i 's/^EXACT_PRE = {"p17352", "p17641"}/EXACT_PRE = {"p17641", "p17352"}/' $SB/build_samdata_ft.py; cp $SB/build_samdata_ft.py $X/edited.py
r=$(run $S $J); [ $r -eq 1 ] && cmp -s $SB/build_samdata_ft.py $X/edited.py && cmp -s $SB/samdata_ft.tpl.html $PR/samdata_ft.tpl.html && ! ls $SB | grep -q bak-monoft \
  && ok "missing anchor: rc 1, no change, no backups ($(head -c 120 $X/sb.out))" || { bad "missing anchor rc $r"; cat $X/sb.out; }
reset; r=$(run 17641 aloev2pm17641); [ $r -eq 1 ] && same_as $PR && ok "S colliding with an existing id (p17641): refused, no change" || { bad "collision rc $r"; cat $X/sb.out; }
rm -rf $SB $PR $X/ref17900 $X/edited.py $X/reg.pyc

# 6 live untouched
[ "$(sha1sum $K/build_samdata_ft.py $K/samdata_ft.tpl.html)" == "$LIVE0" ] && ! ls -a $K | grep -q 'bak-monoft\|\.reg-' && ok "live builder + template unchanged, no backups/temps in klweb" || bad "LIVE FILES CHANGED"
echo "FAILS=$FAILS"

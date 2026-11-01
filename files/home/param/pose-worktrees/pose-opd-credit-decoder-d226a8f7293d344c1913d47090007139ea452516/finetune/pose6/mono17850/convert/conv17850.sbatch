#!/bin/bash
#SBATCH -J c1-conv17850 -p all -w slinky-18 -N1 -n1 -c 2 --mem=24G --nice=0 -t 0-03:00:00 --gres=none
#SBATCH -o /data/home/amartyavarma/ppm/pose6/mono17850/convert/logs/conv17850-%j.log
# C1: 8,1,1 dcp-17850 (HF InternalCan/ppm-v1-c, pinned revision; model tensors only, by range reads) -> ppm-local-bf16-shards-v1
# /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850, written as local-17850.partial, renamed only after the gate; COMPLETE last
# (the stock ckpt_local_export.py --finalize). Gate: all 129 files' sha256 == the 10-22 monoft conversion manifests of the same save.
# Then verify (2): verify_local.py vs local-17641 + kit preflight --mods none. Writes nothing else; deletes nothing.
set -uo pipefail
D=/data/home/amartyavarma/ppm/pose6/mono17850/convert; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850; P=$OUT.partial; ST=$D/conv17850_stats.json
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
echo "== [$(pt)] job $SLURM_JOB_ID on $(hostname -s), cpus $SLURM_CPUS_PER_TASK; /data $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB free"
[ -e "$OUT" ] && { echo "ABORT: $OUT exists (never overwritten)"; exit 1; }
[ -e "$P" ] && { echo "ABORT: $P exists (inspect first)"; exit 1; }
$PY -I -u $D/hf_local_export.py --revision c9b2213751cb3f2ede1dce2c5b9ed275b5cc3ca2 --tag dcp-17850 --ranks all --rep 1 --meta 1 --out "$P" \
  --manifest-dir /data/home/amartyavarma/ppm/aloe-v2pure-main-17850/initctl --stats $ST --threads 16 --chunk-mb 32 2>&1 | grep -v -E "TypedStorage is deprecated|return \(\"STYPE\""
rc=${PIPESTATUS[0]}; echo "== [$(pt)] convert rc $rc"; [ $rc = 0 ] || exit 1
$PY -I -c "
import json, sys
s = json.load(open('$ST')); it = s['items']
names = ['rank%d.pt' % r for r in range(128)] + ['rep.pt']
miss = [n for n in names if n not in it]
bad = [n for n in names if n in it and not (it[n].get('manifest_sha_equal') and it[n].get('manifest_size_equal'))]
rows = [n for n in names[:-1] if n in it and it[n].get('manifest_rows_equal') is not True]
print('GATE files', len(it), 'missing', miss[:5], 'sha/size != 10-22 manifest', bad[:5], 'rows !=', rows[:5])
sys.exit(1 if (miss or bad or rows) else 0)" || { echo "ABORT: gate failed; $P left for inspection (not renamed, no COMPLETE)"; exit 1; }
mv "$P" "$OUT" || exit 1
$PY -I /data/home/amartyavarma/ppm/pose5/code-811-s2p/ckpt_local_export.py --out "$OUT" --finalize || { echo "ABORT: stock --finalize failed"; exit 1; }
echo "== [$(pt)] renamed + COMPLETE '$(cat $OUT/COMPLETE)'; verify 2a: verify_local vs local-17641"
$PY -I -u $D/verify_local.py --new "$OUT" --ref /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641 --json $D/verify_local17850.json
v=$?
echo "== [$(pt)] verify 2b: kit preflight --mods none"
$PY -I -u /data/home/amartyavarma/ppm/pose6/kit/tools/preflight.py --base "$OUT" --mods none --code /data/home/amartyavarma/ppm/pose6/code \
  --json $D/preflight-local-17850.json 2>&1 | grep -v -E "TypedStorage is deprecated" | tail -3
p=${PIPESTATUS[0]}
echo "== [$(pt)] verify_local rc $v, preflight rc $p; /data $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB free; $(du -sBG $OUT | cut -f1)"
if [ $v != 0 ] || [ $p != 0 ]; then mv "$OUT/COMPLETE" "$OUT/COMPLETE.verify-failed"; echo "VERIFY FAILED: COMPLETE moved aside"; exit 1; fi
echo "== [$(pt)] local-17850 READY"

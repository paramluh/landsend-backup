#!/bin/bash
# gcs_backup_ckpt.sh <export_dir>
#   Back up ONE complete checkpoint export dir (.../<run>/w-N) to
#   gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/<run>/<w-N>/ and verify every file.
#   Same machinery as the hfb GCS path (setup-logs/hfb/gcs_up.py): resumable upload (objects already on GCS with the local
#   size + md5 + crc32c are skipped), crc32c-validated in flight, then server-side md5 + crc32c vs the local read for EVERY file,
#   plus a manifest at .../ckpt-backup/<run>/_backup/MANIFEST.<w-N>.tsv (path, size, md5, crc32c, source).
#   On PASS writes <export_dir>/GCS_BACKUP.json (uris, sizes, md5, crc32c, verified, time) and appends a line to
#   setup-logs/hfb/records.jsonl + ppm/qqa-ft/logs/gcs_backup.log. Exits non-zero on any mismatch / missing object / upload error.
#   Sources are only read; nothing is deleted anywhere.
#
#   Called outside Slurm it submits itself as a CPU job on slinky-4 (sbatch -p all -w slinky-4 -c 4 --mem=16G --gres=none)
#   and prints the job id; GCS_BK_WAIT=1 makes it block (srun) and return the job's exit code instead.
#   Key: GOOGLE_APPLICATION_CREDENTIALS by path only (never printed).
set -u
SELF=/data/home/amartyavarma/ppm/qqa-ft/gcs_backup.sh
LOGD=/data/home/amartyavarma/ppm/qqa-ft/logs/gcs_backup
[ $# -eq 1 ] || { echo "usage: $0 <export_dir>" >&2; exit 2; }
E=$(realpath -e "$1") || { echo "no such dir: $1" >&2; exit 2; }
TAG=$(basename "$E"); RUN=$(basename "$(dirname "$E")")
mkdir -p $LOGD
if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "${GCS_BK_WAIT:-0}" = 1 ]; then
    exec srun -J gcsbk-$RUN-$TAG -p all -w slinky-4 -N1 -n1 -c 4 --mem=16G --gres=none -t 1-00:00:00 \
      --output=$LOGD/$RUN-$TAG.log --open-mode=append $SELF "$E"
  fi
  exec sbatch --parsable -J gcsbk-$RUN-$TAG -p all -w slinky-4 -N1 -n1 -c 4 --mem=16G --gres=none -t 1-00:00:00 \
    -o $LOGD/$RUN-$TAG.log --open-mode=append $SELF "$E"
fi

export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json GCS_KEY=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json
export PYTHONUNBUFFERED=1 GCS_HASH_THREADS=4 GCS_MULT=2
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
UP=/data/home/amartyavarma/setup-logs/hfb/gcs_up.py
BUCKET=podcast-raw-audio-data-mass; PREFIX=amartyavarma/ckpt-backup/$RUN
SUMS=/scratch/amartyavarma/gcsbk/sums/$RUN; VOUT=$LOGD/verify/$RUN; mkdir -p $SUMS $VOUT   # per run: tags (w-N) repeat across runs
ML=/data/home/amartyavarma/ppm/qqa-ft/logs/gcs_backup.log
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M PT'; }
echo "[$(pt)] $(hostname -s) job $SLURM_JOB_ID: $E -> gs://$BUCKET/$PREFIX/$TAG/"

[ -f "$E/COMPLETE" ] || { echo "ABORT: $E/COMPLETE missing (export not complete)"; exit 4; }
SPEC=$SUMS/spec-$RUN-$TAG.json
python3 -c "import json,sys; json.dump({'dirs': [[sys.argv[1], sys.argv[2]]], 'exclude': ['GCS_BACKUP.json']}, open(sys.argv[3], 'w'))" "$E" "$TAG" "$SPEC"
rm -f "$E/GCS_BACKUP.json.tmp"

# upload (resumable) + full verify; gcs_up.py rc: 0 PASS, 1 upload errors, 3 verify FAIL, 4 --require vanished
nice -n 10 $PY -W ignore $UP --spec $SPEC --bucket $BUCKET --prefix $PREFIX --tag $TAG --workers 4 \
  --require "$E/COMPLETE" --manifest-name MANIFEST.$TAG.tsv --verify-out $VOUT --sums-dir $SUMS
rc=$?
H=$(hostname -s); V=$VOUT/GCS__${TAG}__$H.json
if [ $rc -ne 0 ]; then
  echo "[$(pt)] FAIL rc $rc ($V)"; echo "$(pt) gcs-backup: $RUN/$TAG FAILED rc $rc (see $V)" >> $ML; exit $rc
fi

# independent re-check against a fresh GCS listing + write the record next to the export
$PY -I -W ignore - "$E" "$BUCKET" "$PREFIX" "$TAG" "$SUMS/$TAG-$H.json" "$V" "$SLURM_JOB_ID" <<'EOF'
import base64, hashlib, json, os, sys, time
import google_crc32c
from google.cloud import storage
E, BUCKET, PREFIX, TAG, SUMF, V, JOB = sys.argv[1:8]
sums = json.load(open(SUMF))
c = storage.Client.from_service_account_json(os.environ["GCS_KEY"])
rem = {b.name: b for b in c.list_blobs(BUCKET, prefix=f"{PREFIX}/{TAG}/")}
files, bad = [], []
for root, dirs, fs in os.walk(E):
    for f in fs:
        p = os.path.join(root, f); rel = os.path.relpath(p, E)
        if rel == "GCS_BACKUP.json" or rel.endswith(".tmp"):
            continue
        st = os.stat(p); k = f"{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}"
        s = sums.get(k)
        if s is None:                                   # not in the upload cache (should not happen): hash now
            m, cc = hashlib.md5(), google_crc32c.Checksum()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(16 << 20), b""):
                    m.update(b); cc.update(b)
            s = {"size": st.st_size, "md5": base64.b64encode(m.digest()).decode(), "crc32c": base64.b64encode(cc.digest()).decode()}
        o = f"{PREFIX}/{TAG}/{rel}"; b = rem.get(o)
        ok = b is not None and b.size == s["size"] and b.md5_hash == s["md5"] and b.crc32c == s["crc32c"]
        if not ok:
            bad.append(rel)
        files.append({"path": rel, "uri": f"gs://{BUCKET}/{o}", "size": s["size"], "md5": s["md5"], "crc32c": s["crc32c"], "verified": ok})
rec = {"export_dir": E, "gcs_uri": f"gs://{BUCKET}/{PREFIX}/{TAG}/", "manifest": f"gs://{BUCKET}/{PREFIX}/_backup/MANIFEST.{TAG}.tsv",
       "files": len(files), "bytes": sum(f["size"] for f in files), "verified": bool(files) and not bad, "mismatches": bad,
       "checks": "every file: GCS object exists, size + md5 + crc32c (server-side) == local read", "gcs_up_verify": V,
       "job": JOB, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "time_pt": __import__("datetime").datetime.now(__import__("zoneinfo").ZoneInfo("America/Los_Angeles")).strftime("%m-%d %H:%M PT"), "file_list": sorted(files, key=lambda f: f["path"])}
open(os.path.join(E, "GCS_BACKUP.json.tmp"), "w").write(json.dumps(rec, indent=1))
os.replace(os.path.join(E, "GCS_BACKUP.json.tmp"), os.path.join(E, "GCS_BACKUP.json"))
open("/data/home/amartyavarma/setup-logs/hfb/records.jsonl", "a").write(json.dumps({"kind": "ckpt-backup", "step": TAG, "job": JOB, "dest": "gcs",
    "path": rec["gcs_uri"], "src": E, "files": rec["files"], "bytes": rec["bytes"], "PASS": rec["verified"], "time": rec["time"]}) + "\n")
print(("PASS" if rec["verified"] else "FAIL"), rec["files"], "files", round(rec["bytes"] / 1e9, 1), "GB", rec["gcs_uri"], bad[:5])
sys.exit(0 if rec["verified"] else 3)
EOF
rc=$?
if [ $rc -eq 0 ]; then
  echo "[$(pt)] VERIFIED PASS -> $E/GCS_BACKUP.json"
  echo "$(pt) gcs-backup: $RUN/$TAG -> gs://$BUCKET/$PREFIX/$TAG/ verified (every file size+md5+crc32c), record $E/GCS_BACKUP.json" >> $ML
else
  echo "[$(pt)] RECHECK FAIL rc $rc"; echo "$(pt) gcs-backup: $RUN/$TAG RECHECK FAILED rc $rc" >> $ML
fi
exit $rc

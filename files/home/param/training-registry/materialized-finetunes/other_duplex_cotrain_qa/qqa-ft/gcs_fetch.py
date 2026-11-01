"""gcs_fetch.py <gs prefix> <local dir>: download every object under the prefix (parallel), verify md5 against GCS, atomically rename dir."""
import base64, hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
uri, out = sys.argv[1], sys.argv[2]
b_, p_ = uri[5:].split("/", 1)
c = storage.Client.from_service_account_json(os.environ["GCS_KEY"])
blobs = [b for b in c.list_blobs(b_, prefix=p_.rstrip("/") + "/")]
tmp = out + ".tmp"; os.makedirs(tmp, exist_ok=True)
def get(b):
    rel = b.name[len(p_.rstrip("/")) + 1:]
    f = os.path.join(tmp, rel); os.makedirs(os.path.dirname(f), exist_ok=True)
    b.download_to_filename(f)
    m = hashlib.md5()
    with open(f, "rb") as fh:
        for x in iter(lambda: fh.read(16 << 20), b""):
            m.update(x)
    return rel, base64.b64encode(m.digest()).decode() == b.md5_hash
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(get, blobs))
bad = [r for r, ok in res if not ok]
assert not bad and res, ("md5 mismatch", bad[:5])
os.replace(tmp, out)
print("FETCH_OK", len(res), "files", out)

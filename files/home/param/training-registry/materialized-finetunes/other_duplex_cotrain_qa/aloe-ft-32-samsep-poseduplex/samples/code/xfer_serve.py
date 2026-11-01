"""pdsamples host-to-host pull, SOURCE side (one per pose host, main amartyavarma-61 10-25): wait for this host's final export part
(NODE_COMPLETE), serve the export dir over HTTP (threaded) and write <xfer>/<host>.serve = 'host:port'; meanwhile sha256 every file ->
<xfer>/<host>.sha256.json {name: [size, sha256]} (the manifest the pullers check). Exits when <xfer>/PULL_DONE exists or after 45 min.
usage: python -I xfer_serve.py <export dir> <xfer dir> <port>"""
import hashlib, http.server, json, os, socket, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
D, X, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])
H = socket.gethostname().split(".")[0]
t0 = time.time()
while not os.path.isfile(f"{D}/NODE_COMPLETE") or any(n.endswith(".tmp") for n in os.listdir(D) if os.path.isdir(D)):
    if time.time() - t0 > 7200:
        sys.exit("no NODE_COMPLETE in 2 h")
    time.sleep(10)
print(f"{H}: {D} NODE_COMPLETE after {time.time() - t0:.0f}s", flush=True)
names = sorted(n for n in os.listdir(D) if os.path.isfile(f"{D}/{n}") and n != "NODE_COMPLETE")
srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), lambda *a: http.server.SimpleHTTPRequestHandler(*a, directory=D))
threading.Thread(target=srv.serve_forever, daemon=True).start()
open(f"{X}/{H}.serve.tmp", "w").write(f"{H}:{PORT}\n"); os.replace(f"{X}/{H}.serve.tmp", f"{X}/{H}.serve")
def sha(n):
    h = hashlib.sha256()
    with open(f"{D}/{n}", "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return n, [os.path.getsize(f"{D}/{n}"), h.hexdigest()]
with ThreadPoolExecutor(4) as ex:
    man = dict(ex.map(sha, names))
json.dump(man, open(f"{X}/{H}.sha256.json.tmp", "w")); os.replace(f"{X}/{H}.sha256.json.tmp", f"{X}/{H}.sha256.json")
print(f"{H}: serving {len(names)} files on :{PORT}, manifest written ({time.time() - t0:.0f}s)", flush=True)
while not os.path.isfile(f"{X}/PULL_DONE") and time.time() - t0 < 2700 + 7200:
    time.sleep(5)
srv.shutdown(); print(f"{H}: done", flush=True)

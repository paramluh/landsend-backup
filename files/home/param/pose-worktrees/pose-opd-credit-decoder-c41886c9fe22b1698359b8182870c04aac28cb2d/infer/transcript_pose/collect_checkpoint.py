#!/usr/bin/env python3
"""Collect one node-local training export without changing its source files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="Export path present on each training host")
    ap.add_argument("destination", type=Path)
    ap.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh/id_ed25519_cluster")
    a = ap.parse_args()
    source, dest = a.source.resolve(), a.destination.resolve()
    if source == dest:
        ap.error("Destination must differ from the training export")
    meta = json.loads((source / "meta.json").read_text())
    assert meta["format"] == "ppm-local-bf16-shards-v1"
    assert int((source / "COMPLETE").read_text()) == meta["step"]
    hosts = meta["hosts"]
    assert len(hosts) == meta["world"]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE").exists():
        raise RuntimeError("Destination is already complete; choose a fresh directory")
    ssh = ["ssh", "-i", str(a.ssh_key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    manifest = {}

    def collect(host):
        names = [f"rank{rank}.pt" for rank, h in enumerate(hosts) if h == host]
        if host == hosts[0]:
            names += ["rep.pt", "meta.json"]
        local = host == socket.gethostname().split(".")[0]
        paths = [source / name for name in names]
        if local:
            sums = {p.name: sha256(p) for p in paths}
            for p in paths:
                target = dest / p.name
                if not target.exists() or sha256(target) != sums[p.name]:
                    shutil.copy2(p, target)
        else:
            command = "sha256sum -- " + " ".join(shlex.quote(str(p)) for p in paths)
            result = subprocess.check_output(ssh + [host, command], text=True)
            sums = {Path(line.split(maxsplit=1)[1].strip()).name: line.split()[0]
                    for line in result.splitlines()}
            subprocess.run(["rsync", "-a", "--checksum", "--partial", "--protect-args", "-e", shlex.join(ssh),
                            "--files-from=-", f"{host}:{source}/", str(dest) + "/"],
                           input="\n".join(names) + "\n", text=True, check=True)
        for name, expected in sums.items():
            assert sha256(dest / name) == expected, (host, name, "checksum mismatch")
        print(f"Verified {host}: {len(names)} files", flush=True)
        return sums

    with ThreadPoolExecutor(max_workers=7) as pool:
        for sums in pool.map(collect, dict.fromkeys(hosts)):
            manifest.update(sums)
    expected = {f"rank{r}.pt" for r in range(meta["world"])} | {"rep.pt", "meta.json"}
    assert set(manifest) == expected
    assert json.loads((dest / "meta.json").read_text()) == meta
    (dest / "SHA256SUMS").write_text("".join(f"{manifest[n]}  {n}\n" for n in sorted(manifest)))
    (dest / "COMPLETE").write_text(f"{meta['step']}\n")
    print(f"Complete: {dest} ({len(manifest)} verified files)", flush=True)


if __name__ == "__main__":
    main()

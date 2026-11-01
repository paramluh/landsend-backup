#!/usr/bin/env python3
"""RELEASE: verify every file the server loads against the pins, using the paths of aloe.env (source it first). Offline; sha256 only.
  python verify_assets.py [--skip-weights]      -> one line per group + PASS / FAIL (exit 0 / 1)
  python verify_assets.py --record-shipped      (release build only: writes the 'shipped' section of env/ASSETS_PINNED.json)
Groups: weights (weights/SHA256SUMS), shipped assets (presets bank, quiet bank, Voxtral audio tower + configs), Mimi + FA3 (pinned public
revisions), Qwen3-32B tokenizer / config (shipped in code/models/Qwen3-32B)."""
import hashlib, json, os, sys
E = os.environ
H = E["ALOE_HOME"]
PIN = f"{H}/env/ASSETS_PINNED.json"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(32 << 20), b""):
            h.update(b)
    return h.hexdigest()

def shipped_files():
    out = {}
    for d, var in (("presets", "ALOE_PRESET_BANK"), ("voxtral", "VOXTRAL_DIR")):
        root = E[var]
        for r, ds, fs in os.walk(root, followlinks=True):
            for f in fs:
                p = os.path.join(r, f)
                out[f"{d}/{os.path.relpath(p, root)}"] = p
    out["quiet_bank/quiet_bank_m2.npz"] = E["ALOE_QUIET_BANK"]
    return out

pin = json.load(open(PIN))
if "--record-shipped" in sys.argv:
    pin["shipped"] = {k: {"size": os.path.getsize(p), "sha256": sha256(p)} for k, p in sorted(shipped_files().items())}
    json.dump(pin, open(PIN, "w"), indent=1)
    print("recorded", len(pin["shipped"]), "shipped files"); sys.exit(0)
res = {}
def check(name, items):
    bad = [(k, "missing") for k, p, _ in items if not os.path.isfile(p)]
    bad += [(k, "sha256") for k, p, want in items if os.path.isfile(p) and sha256(p) != want]
    res[name] = (len(items), bad)
    print(f"{name:28s} {len(items):3d} files  {'OK' if not bad else 'FAIL ' + str(bad[:5])}", flush=True)
if "--skip-weights" not in sys.argv:
    w = E["ALOE_WEIGHTS"]; items = []
    for ln in open(f"{w}/SHA256SUMS"):
        if ln.startswith("#") or not ln.strip(): continue
        h, sz, n = ln.split(None, 2); items.append((n.strip(), f"{w}/{n.strip()}", h))
    check("weights (SHA256SUMS)", items)
sh = shipped_files()
check("shipped assets", [(k, sh.get(k, "/nonexistent/" + k), v["sha256"]) for k, v in pin["shipped"].items()])
A = pin["assets"]
check("voxtral configs (pinned)", [(f, os.path.join(E["VOXTRAL_DIR"], f), v["sha256"]) for f, v in A["voxtral_configs"]["files"].items() if f != "README.md"])
check("voxtral audio tower", [("audio_tower.safetensors", os.path.join(E["VOXTRAL_DIR"], "audio_tower.safetensors"), A["voxtral_audio_tower"]["sha256"])])
check("mimi (kyutai/mimi pinned)", [(f, os.path.join(E["ALOE_MIMI_DIR"], f), v["sha256"]) for f, v in A["mimi"]["files"].items() if f != "README.md"])
check("fa3 (vllm-flash-attn3 v1)", [(f, os.path.join(E["ALOE_FA3_DIR"], f), v["sha256"]) for f, v in A["fa3"]["files"].items()])
check("qwen3-32b tokenizer/config", [(f, os.path.join(E["ALOE_QWEN_DIR"], f), v["sha256"]) for f, v in A["qwen3_32b_tokenizer"]["files"].items()])
ok = all(not b for _, b in res.values())
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

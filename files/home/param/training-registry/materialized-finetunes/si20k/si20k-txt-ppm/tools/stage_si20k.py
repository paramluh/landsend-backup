"""si20k-vox PRE-FLIGHT / STAGING (one task per training node, CPU): (1) copy the 4 si20k stores /data -> node-local (skip a store whose
NODE.json + file sizes already match), verify each staged store's digest == STORES.json and its voice-clip / A-ledger sidecar sha == NODE.json;
(2) build the 64-rank plan from the staged stores (data_si20k_vox.make_plan, digest == STORES.json's); (3) for every logical node this host
runs: every item its 8 ranks read (train incl. the dummy fallback + eval) -> the A ledger entry -> the TTS FLAC: byte size + mtime_ns ==
the ledger, header stereo 24 kHz with >= the row's samples; writes <ledger dir>/DONE_si20k_node<l>.json (the trainer asserts it).
usage: python -I stage_si20k.py --lnodes 0[,1..] [--src DIR] [--dst DIR] [--ledger-dir DIR] [--world 64] [--min-scratch-gb N]"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

CODE = os.environ.get("SI20K_CODE", "/data/home/amartyavarma/ppm/si20k-txt-ppm/code")   # si20k-txt copy (stores read in place when --dst == --src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lnodes", required=True); ap.add_argument("--world", type=int, default=64)
    ap.add_argument("--src", default=os.environ.get("ALOE_SI20K_SRC", "/data/home/amartyavarma/qwen-1ch-data/si20k-vox/stores"))
    ap.add_argument("--dst", default=os.environ.get("ALOE_SI20K_STORE_ROOT", "/scratch/amartyavarma/si20k-vox/stores"))
    ap.add_argument("--ledger-dir", default=os.environ.get("ALOE_SI20K_LEDGER_DIR", "/scratch/amartyavarma/si20k-vox/_ledger"))
    ap.add_argument("--min-scratch-gb", type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.time()
    os.environ["ALOE_SI20K_STORE_ROOT"] = a.dst
    os.environ["ALOE_SI20K_STORES_JSON"] = f"{a.src}/STORES.json"
    os.environ.setdefault("PPM_CD", "2")
    sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/lib")
    import s3rows as F
    sj = json.load(open(f"{a.src}/STORES.json"))
    need = sum(v["bytes"] for v in sj["nodes"].values()) / 1e9
    os.makedirs(a.dst, exist_ok=True)
    free = shutil.disk_usage(a.dst).free / 1e9
    for k in range(4):
        s, d = f"{a.src}/s20n{k}", f"{a.dst}/s20n{k}"
        same = os.path.isfile(f"{d}/NODE.json") and open(f"{s}/NODE.json", "rb").read() == open(f"{d}/NODE.json", "rb").read() and \
            all(os.path.getsize(f"{s}/{f}") == os.path.getsize(f"{d}/{f}") for f in os.listdir(s) if os.path.isfile(f"{d}/{f}")) and \
            sorted(os.listdir(s)) == sorted(os.listdir(d))
        if not same:
            assert free > need + a.min_scratch_gb, ("node-local scratch too small for the stores + exports", free, need, a.min_scratch_gb)
            shutil.rmtree(d, ignore_errors=True)
            subprocess.run(["cp", "-r", s, d + ".tmp"], check=True)
            os.replace(d + ".tmp", d)
        node = F.load_node(d)
        m = json.load(open(f"{d}/NODE.json"))
        assert node["digest"] == sj["nodes"][f"s20n{k}"]["node_digest"], ("staged store digest != STORES.json", d)
        for f_, key in (("voiceclip_B.npz", "voiceclip_B_sha256"), ("A_LEDGER.json", "a_ledger_sha256")):
            assert hashlib.sha256(open(f"{d}/{f_}", "rb").read()).hexdigest()[:16] == m[key] == sj["nodes"][f"s20n{k}"][key], (d, f_)
        print(f"[stage] {d}: {'copied' if not same else 'already staged'}, digest {node['digest']} ok ({time.time() - t0:.0f}s)", flush=True)
    import data_si20k_vox as A
    import soundfile as sf
    PL = A.make_plan(a.world)
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    assert dg == sj["plan"]["digest"] and PL["steps"] == sj["plan"]["steps"], ("plan from the staged stores != STORES.json", dg, sj["plan"]["digest"])
    st = A.Stores()
    os.makedirs(a.ledger_dir, exist_ok=True)
    for l in [int(x) for x in a.lnodes.split(",") if x != ""]:
        its = A.node_items(PL, l, a.world)
        seen, bad = {}, []
        for k, r, _ in its:
            node, rows_, ps, vc, led = st.open(int(k))
            rk = str(vc["row_key"][int(r)])
            e = led[rk]
            if e["flac"] not in seen:
                try:
                    s_ = os.stat(e["flac"]); inf = sf.info(e["flac"])
                    ok = s_.st_size == int(e["flac_bytes"]) and s_.st_mtime_ns == int(e["flac_mtime_ns"]) and inf.channels == 2 and \
                        inf.samplerate == 24000 and inf.frames == int(e["flac_samples"])
                except Exception as ex:  # noqa: BLE001
                    ok = False; e = dict(e, err=repr(ex)[:200])
                seen[e["flac"]] = ok
                if not ok:
                    bad.append(e)
            assert int(e["flac_samples"]) >= (int(e["frame0"]) + int(e["n_frames"])) * 1920 - 6000, ("row past the FLAC end", rk)
        out = {"node": l, "plan_digest": dg, "n_items": len(its), "n_flac": len(seen), "bad": bad[:20], "n_bad": len(bad), "ok": not bad,
               "stores_json_sha": hashlib.sha256(open(f"{a.src}/STORES.json", "rb").read()).hexdigest()[:16], "time": time.strftime("%F %T")}
        json.dump(out, open(f"{a.ledger_dir}/DONE_si20k_node{l}.json.tmp", "w"), indent=1)
        os.replace(f"{a.ledger_dir}/DONE_si20k_node{l}.json.tmp", f"{a.ledger_dir}/DONE_si20k_node{l}.json")
        print(f"[stage] logical node {l}: {len(its)} items, {len(seen)} A FLACs verified (size + mtime + header), bad {len(bad)} ({time.time() - t0:.0f}s)", flush=True)
        assert not bad, ("A FLACs changed / unreadable since the build", bad[:3])
    pp = os.environ.get("SI20K_PODCAST_PLAN", "") if os.environ.get("SI20K_PODCAST", "0") == "1" else ""
    if pp:                                                         # co-training: this host's pretraining slice(s) + every local rank's rows
        import socket
        import podcast_rows as PR
        D = A.D
        host = socket.gethostname().split(".")[0]
        pl = PR.load_plan(pp)
        assert host in pl["hosts"], ("host not in the podcast plan (place the run on the plan's hosts)", host, sorted(pl["hosts"]))
        hp, mode = PR.host_entry(pl, host)                         # native slice if complete on this node, else the staged shards
        print(f"[stage] podcast rows on {host}: {mode} ({hp['native_root']})", flush=True)
        tr, ev = PR.cfgs(pl)
        S = int(os.environ.get("QQA_STEPS", "0"))
        for j in range(8):
            src = PR.Source(pl, host, j, tr)
            assert len(src.codes) >= S, ("not enough unseen podcast rows", host, j, len(src.codes), S)
            m = src.micro(src.codes[0]); me = PR.Source(pl, host, j, ev).micro(src.eval_codes[0]) if src.eval_codes else None
            print(f"[stage] podcast rank {j} = pretraining rank {src.prod_rank}: {len(src.codes)} unseen codes (need {S}), first row "
                  f"{json.loads(m['prov'])['row_key']} T {len(m['pid'])}; eval rows {len(src.eval_codes)}"
                  + (f" (first T {len(me['pid'])})" if me else ""), flush=True)
        man = PR.manifest()                                        # every podcast row this host's ranks read: in the FLAC manifest, FLAC on disk
        miss = []
        for j in range(8):
            src = PR.Source(pl, host, j, tr)
            ds = D.MixedSpeechDS(src.root, src.hp, [], tr, src.prod_rank)
            for code in src.codes[:S] + src.eval_codes:
                slot, r = D.split_code(code)
                rk = ds._open(slot)[1][r]["row_key"]
                e = man.get(rk)
                if e is None or not os.path.isfile(PR._ent(e, "path", "flac")):
                    miss.append(rk)
        print(f"[stage] podcast FLACs on {host}: {8 * S} train + eval rows checked, missing {len(miss)} {miss[:3]}", flush=True)
        assert not miss, ("podcast FLACs not staged for this host's rows", len(miss), miss[:3])
        print(f"[stage] podcast plan {pp} ok on {host} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

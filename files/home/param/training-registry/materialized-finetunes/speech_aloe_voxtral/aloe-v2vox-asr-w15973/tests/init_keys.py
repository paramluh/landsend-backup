"""aloe-v2vox-asr-w15973 pre-launch init check (CPU, compute node): PPM w-15973 vs w-14503 (25963's init) DCP .metadata key sets + shapes + dtypes,
storage files complete; optional: ASR adapter export (argv[1]) keys / shapes / dtype / train_state norm + sha256(bf16) = train_asr_ppm.adapter_sha."""
import hashlib, json, os, pickle, sys
import torch
import torch.distributed.checkpoint as dcp
NEW, REF = "/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973", "/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-14503"
def meta(p):
    md = pickle.load(open(f"{p}/.metadata", "rb"))
    need = {}
    for st in md.storage_data.values():
        need[st.relative_path] = max(need.get(st.relative_path, 0), st.offset + st.length)
    short = [(f, n) for f, n in need.items() if not os.path.isfile(f"{p}/{f}") or os.path.getsize(f"{p}/{f}") < n]
    keys = {k: (tuple(v.size), str(v.properties.dtype)) for k, v in md.state_dict_metadata.items() if hasattr(v, "size")}
    return keys, need, short
kn, nn, sn = meta(NEW)
if os.path.isdir(REF):
    kr, nr, sr = meta(REF)
else:   # w-14503 no longer on /data: 25963's precheck logged 2150 keys, 1432 non-pose, 0 pose_head, 2 decp_bb (pose keys skipped by the trainer)
    print(f"[init] {REF} gone from /data: comparing counts with 25963's precheck (2150 keys, 1432 non-pose)")
    kr = dict(kn)
print(f"[init] w-15973 COMPLETE={open(NEW + '/COMPLETE').read().strip()} storage files {len(nn)} short/missing {len(sn)} {sn[:3]}; {sum(nn.values()) / 1e9:.1f} GB")
print(f"[init] keys w-15973 {len(kn)} vs w-14503 {len(kr)}; only new {sorted(set(kn) - set(kr))[:10]}; only ref {sorted(set(kr) - set(kn))[:10]}")
bad = [k for k in set(kn) & set(kr) if kn[k] != kr[k]]
print(f"[init] shape/dtype mismatches {len(bad)} {bad[:5]}; dtypes {sorted(set(v[1] for v in kn.values()))}")
print(f"[init] pose-prefix keys: " + ", ".join(f"{p} {sum(k.startswith(p) for k in kn)}" for p in ("pose_head.", "tower_p.", "dec_p.", "decp_bb.", "inp.pose.")))
POSE = ("pose_head.", "tower_p.", "dec_p.", "decp_bb.", "inp.pose.")
print(f"[init] w-15973 non-pose keys {sum(not k.startswith(POSE) for k in kn)} (25963 precheck: 1432), total {len(kn)} (2150)")
print(f"[init] VERDICT {'PASS' if not sn and set(kn) == set(kr) and not bad else 'FAIL'}")
if len(sys.argv) > 1:
    p = sys.argv[1]
    md = pickle.load(open(f"{p}/.metadata", "rb")).state_dict_metadata
    ks = sorted(k for k in md if k.startswith("vadapter."))
    sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in ks}
    dcp.load(sd, checkpoint_id=p, no_dist=True)
    ts = json.load(open(f"{p}/train_state.json"))
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].to(torch.bfloat16)
        h.update(k.encode()); h.update(t.contiguous().view(torch.uint8).numpy().tobytes() if t.dim() else t.reshape(1).view(torch.uint8).numpy().tobytes())
    print(f"[adapter] {p}: step {ts['step']} run {ts['cfg'].get('run_tag')} vadapt_norm {ts['cfg'].get('vadapt_norm')} s4_vadapt_norm {ts['cfg'].get('s4_vadapt_norm')} "
          f"enc {ts['cfg'].get('enc_hidden')} stack {ts['cfg'].get('vadapt_stack')} hidden {ts['cfg'].get('vadapt_hidden')} ppm_step {ts['cfg'].get('ppm_step')} ppm_init {ts['cfg'].get('ppm_init')}")
    for k in ks:
        print(f"   {k} {tuple(sd[k].shape)} {sd[k].dtype} rms {sd[k].float().pow(2).mean().sqrt():.6f} finite {bool(torch.isfinite(sd[k].float()).all())}")
    print(f"[adapter] sha256(bf16) {h.hexdigest()[:16]}; target_rms in export: {'vadapter.target_rms' in sd}")

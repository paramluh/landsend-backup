"""D36 unit test (ONE GPU, world 1, synthetic; checkpoint files on node-local /scratch only, deleted at the end).

usage: python tests/test_d36.py <code dir> <scratch dir>      (env: prod_ppm.env sourced; tests/test_d36.sh runs it)
(1) safety_ppm.load_local drop_opt on a toy model (a 'trunk' Linear + a 'pose_head' Linear, AdamW over both, 3 steps, save_local):
    resume into an optimizer WITHOUT the head: weights (head included) restored bit-exact, head has no AdamW state, every other AdamW
    state bit-exact, digest check passes; NEGATIVE: no drop_opt -> fails (strictness kept); drop_opt with the head still in the optimizer
    -> fails; drop_opt naming a parameter without saved state -> fails; a tampered set-aside head exp_avg -> digest mismatch; a tampered
    non-head tensor -> digest mismatch.
(2) train_ppm module-level config (subprocesses, one per env): unset -> no D36 keys, PH_ACTIVE; PPM_POSEHEAD_OFF=1 -> keys
    pose_head_off / pose_head_off_from_step only, PH_OFF; OFF=1 with PPM_POSE_HEAD=0 -> refused; OFF=2 -> refused; pose stats layout
    NV['pose'] shrinks by exactly the head's 3 entries and summarize() emits no pose_head_* keys with the D33 keys at the right offsets."""
import json
import os
import shutil
import subprocess
import sys

CODE, SCR = sys.argv[1], sys.argv[2]
for p in (os.path.join(CODE, "vendor"), os.path.join(CODE, "lib"), CODE):
    sys.path.insert(0, p)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

ok = True


def chk(name, c):
    global ok
    ok &= bool(c)
    print(f"{'PASS' if c else 'FAIL'}  {name}", flush=True)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except AssertionError as e:
        return str(e)[:160]
    return None


# ------------------------------------------------------------------------------------------------ (1) load_local drop_opt
os.environ.update({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29736"})
torch.cuda.set_device(0)
dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
import safety_ppm as SF  # noqa: E402

assert os.path.abspath(os.path.dirname(SF.__file__)) == os.path.abspath(CODE), SF.__file__
dev = torch.device("cuda", 0)
HK = ("pose_head.weight", "pose_head.bias")


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(64, 96)
        self.pose_head = nn.Linear(96, 32)

    def forward(self, x):
        return self.pose_head(self.trunk(x))


def toy(seed):                                                    # FSDP2-sharded like prod (the head its own unit): DTensor params
    from torch.distributed.fsdp import fully_shard
    torch.manual_seed(seed)
    m = Toy().to(dev)
    fully_shard(m.trunk); fully_shard(m.pose_head); fully_shard(m)
    return m


def make(m, with_head):
    ps = list(m.trunk.parameters()) + (list(m.pose_head.parameters()) if with_head else [])
    o = torch.optim.AdamW([{"params": list(m.trunk.parameters()), "lr": 1e-3, "name": "trunk"}]
                          + ([{"params": list(m.pose_head.parameters()), "lr": 2e-3, "name": "pose_head"}] if with_head else []),
                          betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    s = torch.optim.lr_scheduler.LambdaLR(o, [lambda t: 1.0] * len(o.param_groups))
    return o, s, ps


m0 = toy(0)
o0, s0, _ = make(m0, True)
g = torch.Generator(device=dev).manual_seed(1)
for _ in range(3):
    x = torch.randn(16, 64, device=dev, generator=g)
    m0(x).pow(2).mean().backward()
    o0.step(); s0.step(); o0.zero_grad(set_to_none=True)
path = f"{SCR}/toy/dcp-3"
os.makedirs(os.path.dirname(path), exist_ok=True)
SF.save_local(m0, {"opt": o0}, s0, {"step": 3}, path, 0, 0, print, keep=1)
want_w = {n: p.detach().clone() for n, p in m0.named_parameters()}
want_st = {n: {k: v.clone() for k, v in st.items()} for n, st in SF.opt_named_states(m0, o0).items()}

m1 = toy(5)                                                       # different init: everything must come from the file
o1, s1, _ = make(m1, False)
ts = SF.load_local(m1, {"opt": o1}, s1, path, 0, print, drop_opt=HK)
chk("toy params are FSDP2 DTensors (per-rank sharded file, as prod)", all(hasattr(p, "to_local") for p in m1.parameters()))
chk("drop_opt resume: every weight (head included) bit-exact", all(torch.equal(p.to_local(), want_w[n].to_local()) for n, p in m1.named_parameters()))
st1 = SF.opt_named_states(m1, o1)
chk("drop_opt resume: no AdamW state for the head, trunk AdamW state bit-exact",
    set(st1) == {"trunk.weight", "trunk.bias"} and all(torch.equal(st1[n][k], want_st[n][k]) for n in st1 for k in st1[n]))
chk("drop_opt resume: no head parameter in the optimizer", not any(p is q for p in m1.pose_head.parameters() for g_ in o1.param_groups for q in g_["params"]))

m2 = toy(5); o2, s2, _ = make(m2, False)
e = raises(SF.load_local, m2, {"opt": o2}, s2, path, 0, print)
chk(f"NEG no drop_opt -> refused ({e})", e is not None)
m2 = toy(5); o2, s2, _ = make(m2, True)
e = raises(SF.load_local, m2, {"opt": o2}, s2, path, 0, print, drop_opt=HK)
chk(f"NEG drop_opt with the head still in the optimizer -> refused ({e})", e is not None)
m2 = toy(5); o2, s2, _ = make(m2, False)
sd = torch.load(f"{path}/rank0.pt", map_location=dev)
p_bad = f"{SCR}/toy/dcp-3-nostate"
shutil.copytree(path, p_bad)
sd_ = {**sd, "opt": {k: v for k, v in sd["opt"].items() if k != "pose_head.bias"}}
torch.save(sd_, f"{p_bad}/rank0.pt"); open(f"{p_bad}/digest-0.txt", "w").write(SF.digest_state(sd_))
e = raises(SF.load_local, m2, {"opt": o2}, s2, p_bad, 0, print, drop_opt=HK)
chk(f"NEG drop_opt naming a parameter with no saved AdamW state -> refused ({e})", e is not None)
for nm, mut in (("set-aside head exp_avg", lambda d: d["opt"]["pose_head.weight"]["exp_avg"].add_(1e-3)),
                ("non-head trunk exp_avg_sq", lambda d: d["opt"]["trunk.weight"]["exp_avg_sq"].mul_(1.001)),
                ("head weight", lambda d: d["model"]["pose_head.bias"].add_(1e-3))):
    shutil.rmtree(p_bad); shutil.copytree(path, p_bad)
    sd_ = torch.load(f"{path}/rank0.pt", map_location=dev); mut(sd_); torch.save(sd_, f"{p_bad}/rank0.pt")   # digest file = the ORIGINAL
    m2 = toy(5); o2, s2, _ = make(m2, False)
    e = raises(SF.load_local, m2, {"opt": o2}, s2, p_bad, 0, print, drop_opt=HK)
    chk(f"NEG tampered {nm} (digest file untouched) -> digest mismatch ({e})", e is not None and "digest" in e)
dist.destroy_process_group()
shutil.rmtree(f"{SCR}/toy")

# ------------------------------------------------------------------------------------------------ (2) train_ppm config + stats layout
PY = sys.executable
probe = r"""
import sys, json, os
sys.path.insert(0, sys.argv[1])
os.environ.update({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"})
try:
    import train_ppm as T
except AssertionError as e:
    print(json.dumps({"refused": str(e)[:200]})); sys.exit(0)
o = {"keys": sorted(k for k in T.CFG if k.startswith("pose_head")), "off": getattr(T, "PH_OFF", None), "active": getattr(T, "PH_ACTIVE", None),
     "nv_pose": T.NV["pose"], "cfg_off": T.CFG.get("pose_head_off"), "cfg_off_from": T.CFG.get("pose_head_off_from_step")}
v = [float(i + 1) for i in range(T.NV["pose"])]
d = T.summarize("pose", v, "eval")
o["head_keys"] = sorted(k for k in d if "pose_head" in k)
o["decp"] = {k: d[k] for k in sorted(d) if "decp_" in k and k.endswith("_light_ce")}
x = v[len(T.TK):]                                                   # expected: the D33 block right after the pose light stats (+ the head's 3 if active)
o_ = 10 + len(T.CB_S) + len(T.CB_P) + (3 if o["active"] else 0)
o["decp_exp"] = {f"eval_pose/decp_{vn}_light_ce": x[o_ + vi * (2 + len(T.CB_P))] / x[o_ + vi * (2 + len(T.CB_P)) + 1] for vi, vn in enumerate(T.DECP_VARS)}
o["nv_exp"] = len(T.TK) + o_ + 2 * (2 + len(T.CB_P))
print(json.dumps(o))
"""


def run(env_over, code=CODE):
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1"})
    for k, v in env_over.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    r = subprocess.run([PY, "-c", probe, code], env=env, capture_output=True, text=True, cwd="/")
    return json.loads(r.stdout.strip().splitlines()[-1])


on = run({"PPM_POSEHEAD_OFF": None})
off = run({"PPM_POSEHEAD_OFF": "1"})
print("      unset:", on)
print("      OFF=1:", off)
chk("unset: no D36 keys in CFG, PH_ACTIVE, head stats present", on.get("cfg_off") is None and on["active"] is True and on["off"] is False
    and "pose_head_off" not in on["keys"] and len(on["head_keys"]) == 2)
chk("OFF=1: exactly the 2 D36 keys added, PH_OFF, no pose_head_* stats",
    set(off["keys"]) - set(on["keys"]) == {"pose_head_off", "pose_head_off_from_step"} and off["off"] is True and off["active"] is False
    and off["cfg_off"] == 1 and off["head_keys"] == [])
chk("OFF=1: pose stats vector exactly 3 shorter (the head's sum / n / hits)", on["nv_pose"] - off["nv_pose"] == 3)
chk("D33 decp_* eval keys read from the slots right after the pose light stats (+3 only with the head active), NV consistent",
    all(r["decp"] == r["decp_exp"] and len(r["decp"]) == 2 and r["nv_pose"] == r["nv_exp"] for r in (on, off)))
r0 = run({"PPM_POSEHEAD_OFF": "1", "PPM_POSE_HEAD": "0"})
chk(f"OFF=1 with PPM_POSE_HEAD=0 -> refused ({r0.get('refused')})", "refused" in r0)
r2 = run({"PPM_POSEHEAD_OFF": "2"})
chk(f"OFF=2 -> refused ({r2.get('refused')})", "refused" in r2)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)

"""evalvox CONTROL eval env (10-24, main): pure PPM 16861 init + an ASR adapter export, s recalibrated at start (= the FT's step-0 path), eval-only
(ALOE_EVAL_ONLY=1) = run_a16861.env + [ctl] lines. usage: python3 -I mkenv_ctl.py <name> <adapter dir> <sha16>"""
import sys
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep"; name, ad, sha = sys.argv[1:4]
new = {"QQA_RUN": f"evalctl-{name}", "QQA_EXPORT": f"{D}/evalvox/no-export-{name}", "ALOE_LOCAL_EXPORT_ROOT": f"/scratch/amartyavarma/evalvox/no-exports-{name}",
       "VOX_ADAPTER": ad, "VOX_ADAPTER_SHA": sha, "ALOE_OPT_OFFLOAD": "0", "HF_REPO_INTER": "InternalCan/ppm-aloe-evalvox-unused",
       "HF_REPO_FINAL": "InternalCan/ppm-aloe-evalvox-unused", "HFW_NAME": f"evalctl-{name}"}
out = [f"# evalvox CONTROL ({name}): eval-only, PPM 16861 + {ad} (sha {sha}), s recalibrated = run_a16861.env + [ctl] lines", "ALOE_EVAL_ONLY=1   # [ctl]"]
seen = set()
for l in open(f"{D}/run_a16861.env").read().splitlines():
    k = l.split("=", 1)[0] if "=" in l and not l.startswith("#") else None
    if k in new:
        out.append(f"{k}={new[k]}   # [ctl]"); seen.add(k)
    else:
        out.append(l)
assert seen == set(new), set(new) - seen
open(f"{D}/evalvox/env-ctl-{name}.env", "w").write("\n".join(out) + "\n"); print(f"{D}/evalvox/env-ctl-{name}.env")

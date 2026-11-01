"""evalvox: env for the eval-only Aloe eval of one PPMVOX export = run_a16861.env (world 14: slinky-18 x8 + slinky-4 GPUs 0-5, node-local sam-sep
stores, logical world 64) with the [evalvox] lines. usage: python3 -I mkenv.py <tag e.g. w-16911>"""
import sys
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep"; tag = sys.argv[1]; init = f"/scratch/amartyavarma/evalvox/{tag}"
tag = tag.replace("/", "-")                         # an HF-prefixed tag (ppmvox-v2/w-N) -> name ppmvox-v2-w-N (dirs / run name)
new = {"QQA_INIT": init, "QQA_RUN": f"evalvox-ppmvox-{tag}", "QQA_EXPORT": f"{D}/evalvox/no-export-{tag}",
       "ALOE_LOCAL_EXPORT_ROOT": f"/scratch/amartyavarma/evalvox/no-exports-{tag}", "VOX_ADAPTER": init, "VOX_ADAPTER_SHA": "",
       "ALOE_OPT_OFFLOAD": "0", "HF_REPO_INTER": "InternalCan/ppm-aloe-evalvox-unused", "HF_REPO_FINAL": "InternalCan/ppm-aloe-evalvox-unused",
       "HFW_NAME": f"evalvox-{tag}"}
out = [f"# evalvox ({tag}): EVAL-ONLY of the PPMVOX export {tag} (its own vadapter + vgain theta) on the Aloe FT's eval = run_a16861.env + [evalvox] lines",
       "ALOE_EVALVOX=1   # [evalvox]"]
seen = set()
for l in open(f"{D}/run_a16861.env").read().splitlines():
    k = l.split("=", 1)[0] if "=" in l and not l.startswith("#") else None
    if k in new:
        out.append(f"{k}={new[k]}   # [evalvox]"); seen.add(k)
    else:
        out.append(l)
assert seen == set(new), set(new) - seen
open(f"{D}/evalvox/env-{tag}.env", "w").write("\n".join(out) + "\n")
print(f"{D}/evalvox/env-{tag}.env")

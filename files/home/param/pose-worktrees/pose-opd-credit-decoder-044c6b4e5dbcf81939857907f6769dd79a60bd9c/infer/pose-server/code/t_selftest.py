import sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import fast_ppm_srv  # noqa: F401,E402  (same import order as the worker)
import srv_common as C  # noqa: E402
import prompt_build as PB  # noqa: E402
r = PB.selftest(C.clips())
print(r)
print("ALL PASS" if all(ok for _, ok, _ in r) else "FAIL")
p, _ = PB.PR.build_prefix("Persona\nYour name is Mia. You love hiking.", PB.rule_clip(__import__("pickle").load(open(C.clips()[0]["row"] + ".pkl", "rb"))))
print("persona P", p["P"])

"""run verify_dcp_host's selftest (+ a real-file check: rep.pt of the pose-duplex dcp-498 is NOT needed; this only exercises the small cases)."""
import importlib.util, sys, os
sys.argv = ["x", "/nonexistent", "0", "0", "8", "/dev/null"]
spec = importlib.util.spec_from_file_location("vdh", "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools/verify_dcp_host.py")
src = open(spec.origin).read().split("ranks = list(range(")[0]
g = {"__name__": "vdh"}; exec(compile(src, spec.origin, "exec"), g)
import torch
s2 = importlib.util.spec_from_file_location("sp", "/data/home/amartyavarma/ppm/ppmvox/code/safety_ppm.py"); m = importlib.util.module_from_spec(s2); s2.loader.exec_module(m)
print("selftest", g["selftest"](m, torch))

"""CPU test: safety_ppm.prune honours PROTECT_STEPS (8,1,1: never delete the switch checkpoint). Fake dcp dirs in a temp dir."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import safety_ppm as SF  # noqa: E402

root = tempfile.mkdtemp(prefix="ppm811-protect-")
for st in (16735, 16762, 16800, 16850, 16900):
    os.makedirs(f"{root}/dcp-{st}"); open(f"{root}/dcp-{st}/COMPLETE", "w").write(str(st))
SF.PROTECT_STEPS.add(16762)
gone = SF.prune(root, 2, 0, print)
left = SF.complete_steps(root)
ok1 = gone == [16735, 16800] and left == [16762, 16850, 16900]
SF.PROTECT_STEPS.clear()
gone2 = SF.prune(root, 2, 0, print)
ok2 = gone2 == [16762] and SF.complete_steps(root) == [16850, 16900]
print(("PASS" if ok1 else "FAIL"), "protect on: keep-last-2 keeps the switch save dcp-16762, rotates the rest:", gone, "left", left)
print(("PASS" if ok2 else "FAIL"), "protect off (production): keep-last-2 as before:", gone2)
sys.exit(0 if ok1 and ok2 else 1)

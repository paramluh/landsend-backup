#!/usr/bin/env python3
"""pose5 watchdog v3 (audit 4 R4-2): did this trainer job die LOADING its resume save? Prints "LOADFAIL <exception line>" or nothing.
Only a real Python traceback counts (lines "[rankNN]: Traceback (most recent call last):" + that rank's indented frame lines + its exception
line), and only if the job never logged "resumed <dir> at step N" and the traceback runs through the resume load: a frame "in load_local" /
"in load" of safety_ppm.py / train_ppm.py, or torch.load / digest_state, or an exception line naming the resume (digest, COMPLETE,
unpickling, PytorchStreamReader, "checkpoint parameter names differ"). Node / NCCL / OOM / data-loader deaths print nothing.
usage: wd_loadfail.py <job log>"""
import re
import sys

try:
    lines = open(sys.argv[1], errors="replace").read().splitlines()
except OSError:
    sys.exit()
if any(re.search(r"resumed \S+ at step \d+", l) for l in lines):
    sys.exit()
FRAME = re.compile(r'File "[^"]*(safety_ppm|train_ppm)\.py", line \d+, in (load_local|load)\b|torch/serialization\.py|in digest_state\b')
EXC = re.compile(r"digest|resume source not COMPLETE|UnpicklingError|PytorchStreamReader|checkpoint parameter names differ|"
                 r"allow_missing|drop_model|drop_opt|EOFError|unexpected EOF|Ran out of input")
i = 0
while i < len(lines):
    m = re.match(r"^(\[rank\d+\]: |\d+: |)Traceback \(most recent call last\):", lines[i])
    if not m:
        i += 1; continue
    pre, frames, exc = m.group(1), [], ""
    j = i + 1
    while j < len(lines) and lines[j].startswith(pre):
        body = lines[j][len(pre):]
        if body.startswith((" ", "\t")):
            frames.append(body); j += 1; continue
        exc = body; j += 1; break
    if any(FRAME.search(f) for f in frames) or (exc and EXC.search(exc)):
        print("LOADFAIL " + (exc or frames[-1].strip())[:200]); sys.exit()
    i = j

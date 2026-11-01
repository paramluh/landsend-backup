"""PPM main step-time probe: last N step lines of ppm-<job>.log -> mean / max s/step, mean pose/speech/text, max wait."""
import re, sys
job = sys.argv[1] if len(sys.argv) > 1 else "22224"; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
pat = re.compile(r"^\[(\S+ \S+) PT\] step (\d+)/\d+ ([\d.]+)s \(pose ([\d.]+) speech ([\d.]+) text ([\d.]+) wait ([\d.]+)\)")
rows = []
for line in open(f"/data/home/amartyavarma/ppm/logs/ppm-{job}.log", errors="replace"):
    m = pat.match(line)
    if m:
        rows.append((m.group(1), int(m.group(2)), *map(float, m.groups()[2:])))
r = rows[-n:]
mean = lambda i: sum(x[i] for x in r) / len(r)
print(f"PPM {job} steps {r[0][1]}-{r[-1][1]} (to {r[-1][0]} PT): s/step mean {mean(2):.2f} max {max(x[2] for x in r):.1f} | pose {mean(3):.2f} speech {mean(4):.2f} text {mean(5):.2f} | wait max {max(x[6] for x in r):.1f}")

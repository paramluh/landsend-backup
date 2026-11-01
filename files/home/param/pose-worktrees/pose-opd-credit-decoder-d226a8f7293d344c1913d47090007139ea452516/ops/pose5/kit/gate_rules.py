"""pose5 disk GATE RULES: the ONE code path for the pre-pause gate (c/disk/build_disk_need.py, records) and the live gate (kit/stage_host.sh,
after staging), so the two can never disagree on the rules. Units: GB = 1e9 B.
  floor   : slinky-13 20 GB (main 10-27), others 30
  margin  : slinky-13 3.5 GB (main 10-27 fail-safe), others 5
  saves   : max(46.09 live + 53.0 next-save write room (KEEP=1 + chained hold) + floor, 2 x 46.09 + HF watcher FLOOR_GB) -- the running hfb-watch
            holds a new save only if free - save >= FLOOR_GB after it
  window  : npl5 roll window at the host's ahead, back window b110 (B's npl5/roll/WINDOW-<key>-a<ahead>-b110.json; fallback = B's slinky-0
            figures a100 11.70 / a60 9.48 / a40 8.38 GB). Default ahead 100, or with AUTOSHRINK the largest ahead >= 40 that fits.
            WEKA-SERVED hosts (main 10-27 ~10:05, audit 2 R2-2: slinky-13): the npl5 window lives on Weka (/scratch/amartyavarma/npl-append/npl5/
            jobs = symlink to /data/home/amartyavarma/ppm/pose5/npl5-weka/<host>/jobs) -> its /scratch window term is 0 in BOTH gates and its
            ahead is the default 100; the window's GB go to the /data ledger (weka_window_gb)
  evaluate(host, free_gb, pending_gb): left = free_gb - pending_gb - saves - margin; ok = left >= 0.
            pre-pause: free_gb = recorded free + what the switch frees (dcp-17798, approved caches); pending = window + slinky-15 slot-9 extras
            live     : free_gb = df after staging;                         pending = window - npl5 already on disk (extras are on disk)
CLI (stage_host.sh): gate_rules.py ahead <host> <free_gb> <hfb_floor> <autoshrink 0|1>   -> "<ahead> <window_gb>"
                     gate_rules.py eval  <host> <free_gb> <pending_gb> <hfb_floor>       -> "<need_gb> <left_gb> OK|SHORT" (exit 0 | 3)
                     gate_rules.py weka  <host>                                          -> Weka window dir (exit 0) | "" (exit 1)
  fetcher floor (stage_host.sh -> kit/state/MIN_FREE_GB-<host>): WEKA_HOSTS 350 (B's fetcher checks statvfs of the resolved dest = /data, so
            /data stays >= 350; main 10-27), others 53 + floor (room for the trainer's next save on /scratch)"""
import glob, json, sys

SAVE, WRITE = 46.09, 1.15 * 46.09
FALLBACK = {100: 11.70, 80: 10.6, 60: 9.48, 40: 8.38}
EXTRAS_S15 = 9.2 + 10.24 + 0.2                    # slot-9 PPM pose store + B's npl3 rows + plans


def floor(h): return 20.0 if h == "slinky-13" else 30.0
def margin(h): return 3.0 if h == "slinky-13" else 5.0     # s13 3.0 (main 10-27 11:34, after stage 31091 aborted s13 short 0.2 at 3.5)
# saves term vs the chained-hold rule (10-27): v2+ releases held N once N is HF-verified AND N+1 is COMPLETE -> peak = N + the next save's
# write room (SAVE + WRITE + floor), and hfb holds N+1 with only N+1 on disk (2 x SAVE + hfb FLOOR). The v1 rule (release N only after N+1
# is HF-verified) needed 3 x SAVE + hfb FLOOR (= 158.3 GB at FLOOR 20) for hfb to hold N+1 next to held N: the 13:4x deadlock on s6 (141.1).
def saves(h, hfb_floor): return max(SAVE + WRITE + floor(h), 2 * SAVE + float(hfb_floor))
def key(h): return {"slinky-15": "slinky-9"}.get(h, h)
def default_ahead(h): return 100                  # slinky-13 back to 100 (its window is on Weka now; was 40 while it sat on /scratch)
WEKA_HOSTS = {"slinky-13": "/data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-13",   # main 10-27 ~10:05
              "slinky-6": "/data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-6",     # ppmguard (acting main) 10-27 ~10:50: audit 3 R3-1 (was autoshrunk to a40)
              "slinky-12": "/data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-12"}   # main 10-27 11:34: stage 31091 aborted s12 short 0.8 GB
def weka_window_gb(h, ahead=None, root=None):     # /data GB of a Weka-served host's window (0 for the others)
    if h not in WEKA_HOSTS: return 0.0
    w = windows(h, root) if root else windows(h); a = ahead or default_ahead(h)
    return w.get(a, max(w.values()))


def windows(h, root="/data/home/amartyavarma/ppm/pose5/npl5/roll"):
    w = {}
    for f in glob.glob(f"{root}/WINDOW-{key(h)}-a*-b110.json"):
        j = json.load(open(f)); w[int(j["ahead"])] = float(j["peak_window_GB"])
    return w or dict(FALLBACK)


def choose_ahead(h, free_gb, hfb_floor, autoshrink=False, root=None):
    """-> (ahead, /scratch window_gb) given the free GB available BEFORE the window lands (Weka-served hosts: (100, 0.0))"""
    w = windows(h, root) if root else windows(h)
    a0 = default_ahead(h)
    if a0 not in w:
        a0 = max([a for a in w if a <= a0] or [min(w)])
    if h in WEKA_HOSTS:
        return a0, 0.0
    if not autoshrink:
        return a0, w[a0]
    room = free_gb - saves(h, hfb_floor) - margin(h)
    if w[a0] <= room:
        return a0, w[a0]
    fit = [a for a in w if 40 <= a <= a0 and w[a] <= room]
    return (max(fit), w[max(fit)]) if fit else (40, w.get(40, FALLBACK[40]))


def evaluate(h, free_gb, pending_gb, hfb_floor):
    need = saves(h, hfb_floor) + margin(h) + pending_gb
    left = free_gb - need
    return round(need, 1), round(left, 1), left >= 0


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "ahead":
        a, g = choose_ahead(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "1"); print(a, g)
    elif m == "wekahosts":                        # gate_rules.py wekahosts -> the Weka-served hosts, one per line
        print("\n".join(sorted(WEKA_HOSTS)))
    elif m == "weka":                             # gate_rules.py weka <host> -> the host's Weka window dir (exit 0) | nothing (exit 1)
        p = WEKA_HOSTS.get(sys.argv[2]); print(p or ""); sys.exit(0 if p else 1)
    elif m == "eval":
        n, l, ok = evaluate(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])); print(n, l, "OK" if ok else "SHORT"); sys.exit(0 if ok else 3)

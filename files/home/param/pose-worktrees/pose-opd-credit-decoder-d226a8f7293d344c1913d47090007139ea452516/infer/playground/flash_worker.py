"""10-26 (main): the playground's flash + beep burn in its OWN job (genweb/flash.sbatch: 8 CPUs), not in the 2-CPU web server: every second
it touches render/flash_worker.heartbeat and burns every finished exact render that has a prefill switch and no <id>.flash.mp4 yet, newest
first, up to 4 at once, with server.py's own burn_flash() (the same ffmpeg command and settings) under the same per-clip lock
(render/out/<id>.flash.lock, O_EXCL, stale after 10 min). The web server's flasher burns only while this heartbeat is stale (> 30 s)."""
import concurrent.futures as cf
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as S  # noqa: E402

PAR = 4


def todo(busy):
    out = []
    for rid in sorted((r for r in os.listdir(S.RUNS) if S.RID.match(r)), reverse=True)[:400]:
        if rid in busy:
            continue
        src, fl = f"{S.ROUT}/{rid}.mp4", f"{S.ROUT}/{rid}.flash.mp4"
        if os.path.exists(fl) or not os.path.exists(src) or time.time() - S._mt0(src) < 2:
            continue
        t = S.switch_s(S.jread(f"{S.RUNS}/{rid}/request.json", {}))
        if t is not None:
            out.append((rid, src, fl, t))
    return out


def burn(rid, src, fl, t):
    if not S.flash_lock(rid):
        return rid, None, "locked"
    t0 = time.time()
    try:
        ok, err = S.burn_flash(src, fl, t)
    finally:
        S.flash_unlock(rid)
    return rid, round(time.time() - t0, 1), "" if ok else err


def main():
    print(f"[flash] {S.ROUT}, {PAR} in parallel", flush=True)
    busy = {}
    with cf.ThreadPoolExecutor(PAR) as ex:
        while True:
            with open(S.FLASH_HB, "w") as f:
                f.write(str(time.time()))
            for fut in [f for f in busy.values() if f.done()]:
                rid, dt, err = fut.result()
                busy.pop(rid, None)
                if dt is not None:
                    print(f"[flash] {time.strftime('%H:%M:%S')} {rid} {dt} s {err}", flush=True)
            for rid, src, fl, t in todo(busy)[:max(0, PAR - len(busy))]:
                busy[rid] = ex.submit(burn, rid, src, fl, t)
            time.sleep(1)


if __name__ == "__main__":
    main()

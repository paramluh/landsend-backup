#!/usr/bin/env python3
"""Register the monoplex FT run ppm/aloe-v2pure-main-<S> as scaling point p<S> on the samdata_ft page (build_samdata_ft.py + samdata_ft.tpl.html in
setup-logs/klweb; rebuilt every 60 s by job 26082 on slinky-18). Applies exactly the edits of the 10-23 investigation (patch.diff):
  builder:  FT_RUNS += Run("p<S>", <S>, .../aloe-v2pure-main-<S>, "ppm-aloe-v2pure-main-<S>-j", r"^<JOBNAME>$"); SC_ORDER += "p<S>"; EXACT_PRE += "p<S>";
            lr_runs() += ("p<S>", paths.get("p<S>"), "ppm")
  template: --cm colour (light/dark), NM label + PPMS.push([...p<S>...]), PORD gets 'p<S>' first, RUNC['p<S>'], SCR['p<S>'].
The one deliberate difference to the placeholder diff: the NM display label writes S with a thousands comma (17,852), like every other run label.

Usage: python3 samdata_register.py <S> <JOBNAME> [--dry] [--replace]
  (default)  edit the LIVE files: backups <file>.bak-monoft-<S> first, temp file + atomic rename, py_compile the builder, restore both from the
             backups on any failure.
  --dry      write the patched copies to /data/home/amartyavarma/ppm/monoft/dryrun/samdata/<S>/ (the live files are never touched) and print the
             diff. The dry builder copy reads the template next to it (W = its own dir) and writes into its argv[1] (default: its own dir).
  --replace  if a DIFFERENT main point (other S or other job name) is registered, unregister it first (exact reverse of its edits), then register.
Idempotent: the same S + JOBNAME already registered -> no-op (exit 0). Every edit is anchored on the exact current text (each anchor must occur
exactly once); a missing / duplicated anchor or a half-registered state -> fail loudly, no change. Exit: 0 ok / no-op, 1 failure, 2 refused.
Test hooks: SAMDATA_REG_KLWEB=<dir> (operate on a sandbox copy of klweb), SAMDATA_REG_FAILTEST=1 (fail after the writes -> restore path).
Stdlib only; light (file edits + py_compile), fine on the login node."""
import argparse
import difflib
import os
import py_compile
import re
import shutil
import sys
import tempfile

K = os.environ.get("SAMDATA_REG_KLWEB", "/data/home/amartyavarma/setup-logs/klweb")   # test hook: a sandbox copy of klweb
FILES = {"builder": f"{K}/build_samdata_ft.py", "template": f"{K}/samdata_ft.tpl.html"}
DRY_ROOT = "/data/home/amartyavarma/ppm/monoft/dryrun/samdata"


class Fail(Exception):
    pass


def edits(S, JOB):
    """[(file key, old, new)]: anchored exact replacements = patch.diff with <S> / <JOBNAME> filled in. Each `old` is the CURRENT text (unregistered)
    and must occur exactly once; each `new` is unique to this (S, JOB) and is what an applied registration contains."""
    lab = f"{S:,}"
    b_run_old = ('           Run("p17641", 17641, f"{H}/ppm/aloe-v2pure-17641", "ppm-aloe-v2pure-17641-j", r"^aloev2p17641$")]\n'
                 'REF_INIT = 11488   # ')
    b_run_new = ('           Run("p17641", 17641, f"{H}/ppm/aloe-v2pure-17641", "ppm-aloe-v2pure-17641-j", r"^aloev2p17641$"),\n'
                 f'           # 10-23 scaling point 7: init = PPM 8,1,1 delivery-3 append run (job 27653) step {S}; = aloe-v2pure-17641 verbatim except the init\n'
                 f'           Run("p{S}", {S}, f"{{H}}/ppm/aloe-v2pure-main-{S}", "ppm-aloe-v2pure-main-{S}-j", r"^{JOB}$")]\n'
                 'REF_INIT = 11488   # ')
    b_sc_old = 'SC_ORDER = ("p7932", "ref", "p13533", "ft", "p17352", "p17641")   # scaling points, by init step\n'
    b_sc_new = f'SC_ORDER = ("p7932", "ref", "p13533", "ft", "p17352", "p17641", "p{S}")   # scaling points, by init step\n'
    b_ex_old = 'EXACT_PRE = {"p17352", "p17641"}   # 10-22: x must be the pretraining eval at exactly the init step\n'
    b_ex_new = f'EXACT_PRE = {{"p17352", "p17641", "p{S}"}}   # 10-22: x must be the pretraining eval at exactly the init step\n'
    b_lr_old = '             ("p17352", paths.get("p17352"), "ppm"), ("p17641", paths.get("p17641"), "ppm")]\n            + [(k, f"{d}/logs/metrics.jsonl", "sam")'
    b_lr_new = (f'             ("p17352", paths.get("p17352"), "ppm"), ("p17641", paths.get("p17641"), "ppm"), ("p{S}", paths.get("p{S}"), "ppm")]\n'
                '            + [(k, f"{d}/logs/metrics.jsonl", "sam")')
    t_css_old = (':root[data-theme=dark]{--c13:#5fd3c7;--c79:#f06595;--c17:#ffd43b;--c18:#d9a066;--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--muted:#9aa1ad;'
                 '--line:#2a2f3a;--one:#6aa8ff;--pose:#5fd3c7;--sam:#e6e8ec;--s3:#a3aab6;--dm:#ff922b;--he:#e599f7;--evb:#b197fc;--ilp:#8ce99a;--nxt:#faa2c1;'
                 '--btn:#232834;--on:#6aa8ff;--onfg:#0f1115}\n')
    t_css_new = (t_css_old + ':root{--cm:#5c940d}@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--cm:#a9e34b}}:root[data-theme=dark]{--cm:#a9e34b}'
                 f'   /* p{S} (aloe-v2pure-main-{S}) */\n')
    t_css_old += '*{box-sizing:border-box}\n'
    t_css_new += '*{box-sizing:border-box}\n'
    t_pord_old = ("'ppm-aloe-v2pure-17641',17641]];\n"
                  "const PORD=['p17641','p17352','ft','p13533','ref','p7932'];   /* PPM curve order on every chart: init step high -> low (ref = 23051, init 11,488) */\n")
    t_pord_new = ("'ppm-aloe-v2pure-17641',17641]];\n"
                  f"const NM='PPM 32B from step {lab} (8,1,1 + delivery 3)';PPMS.push(['p{S}',NM,'--cm',()=>(S().fts||{{}})['p{S}']||null,'ppm-aloe-v2pure-main-{S}',{S}]);\n"
                  f"const PORD=['p{S}','p17641','p17352','ft','p13533','ref','p7932'];   /* PPM curve order on every chart: init step high -> low (ref = 23051, init 11,488) */\n")
    t_runc_old = ("p17352:()=>[N17,'--c17',ftX],p17641:()=>[N18,'--c18',ftX]};for(const [id,nm,col] of SAMS)RUNC[id]=()=>[nm,col,samX];\n"
                  "const LRDASH=")
    t_runc_new = ("p17352:()=>[N17,'--c17',ftX],p17641:()=>[N18,'--c18',ftX]};for(const [id,nm,col] of SAMS)RUNC[id]=()=>[nm,col,samX];\n"
                  f"RUNC['p{S}']=()=>[NM,'--cm',ftX];\n"
                  "const LRDASH=")
    t_scr_old = ("const SCR={ft:[FTN,'--one'],p13533:[N13,'--c13'],ref:[REFN,'--dm'],p7932:[N79,'--c79'],p17352:[N17,'--c17'],p17641:[N18,'--c18']};\n"
                 "const SCK=")
    t_scr_new = ("const SCR={ft:[FTN,'--one'],p13533:[N13,'--c13'],ref:[REFN,'--dm'],p7932:[N79,'--c79'],p17352:[N17,'--c17'],p17641:[N18,'--c18']};\n"
                 f"SCR['p{S}']=[NM,'--cm'];\n"
                 "const SCK=")
    return [("builder", b_run_old, b_run_new), ("builder", b_sc_old, b_sc_new), ("builder", b_ex_old, b_ex_new), ("builder", b_lr_old, b_lr_new),
            ("template", t_css_old, t_css_new), ("template", t_pord_old, t_pord_new), ("template", t_runc_old, t_runc_new),
            ("template", t_scr_old, t_scr_new)]


def swap(texts, eds, forward, what):
    """Apply (forward: old -> new) or revert (new -> old) every edit; each source string must occur exactly once in the current text."""
    out = dict(texts)
    for i, (fk, old, new) in enumerate(eds if forward else reversed(eds)):
        src, dst = (old, new) if forward else (new, old)
        n = out[fk].count(src)
        if n != 1:
            raise Fail(f"{what}: anchor #{i + 1} in {FILES[fk]} occurs {n}x (need exactly 1) -> files changed since patch.diff? nothing written.\n"
                       f"  anchor starts: {src[:110]!r}")
        out[fk] = out[fk].replace(src, dst, 1)
    return out


MAIN_RUN = re.compile(r'Run\("p(\d+)", (\d+), f"\{H\}/ppm/aloe-v2pure-main-(\d+)", "ppm-aloe-v2pure-main-(\d+)-j", r"\^([^"$]*)\$"\)')
TPL_MARK = re.compile(r"PPMS\.push\(\['p(\d+)',NM,|--cm:|\bconst NM=")


def registered(texts):
    """-> list of (S, JOB) main points fully registered in both files; raises Fail on a half / inconsistent registration."""
    found = []
    for m in MAIN_RUN.finditer(texts["builder"]):
        s = {m.group(1), m.group(2), m.group(3), m.group(4)}
        if len(s) != 1:
            raise Fail(f"inconsistent main Run line in the builder: {m.group(0)!r}")
        found.append((int(m.group(1)), m.group(5)))
    if len(found) > 1:
        raise Fail(f"more than one main-<S> point in the builder ({found}); fix by hand")
    tmarks = TPL_MARK.findall(texts["template"])
    if not found:
        if tmarks:
            raise Fail("half-registered: the template has a main-<S> point (NM / --cm / PPMS.push) but the builder has none; fix by hand")
        return []
    S0, J0 = found[0]
    for fk, _, new in edits(S0, J0):
        if texts[fk].count(new) != 1:
            raise Fail(f"half-registered p{S0} ({J0}): {FILES[fk]} lacks (or duplicates) the edit starting {new[:110]!r}; fix by hand")
    return found


def atomic_write(path, text, mode):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.reg-", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def check_builder(path):
    """py_compile into a throwaway .pyc (no __pycache__ litter next to the live builder)."""
    fd, cf = tempfile.mkstemp(suffix=".pyc")
    os.close(fd)
    try:
        py_compile.compile(path, cfile=cf, doraise=True)
    finally:
        os.unlink(cf)


def check_template(text, S, JOB):
    for _, _, new in [e for e in edits(S, JOB) if e[0] == "template"]:
        if text.count(new) != 1:
            raise Fail("template check: an applied edit is missing after the write")
    for tok in ("__DATA__", "<script>", "</script>"):
        if tok not in text:
            raise Fail(f"template check: {tok!r} missing")


def read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def diff_text(a, b, path):
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{os.path.basename(path)}", f"b/{os.path.basename(path)}", n=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("S", type=int, help="init step = the main run's paused save step (run dir ppm/aloe-v2pure-main-<S>)")
    ap.add_argument("JOBNAME", help="Slurm job name of the FT training job (e.g. aloev2pm<S>); matched as ^JOBNAME$")
    ap.add_argument("--dry", action="store_true", help=f"write patched copies to {DRY_ROOT}/<S>/ and print the diff; never touch the live files")
    ap.add_argument("--replace", action="store_true", help="replace a different registered main point")
    a = ap.parse_args()
    S, JOB = a.S, a.JOBNAME
    if not 1000 <= S <= 10_000_000:
        print(f"refused: S={S} out of range", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", JOB):
        print(f"refused: JOBNAME {JOB!r} must be [A-Za-z0-9_-] (it goes into a regex)", file=sys.stderr)
        return 2
    if JOB != f"aloev2pm{S}":
        print(f"warning: JOBNAME {JOB!r} != the launcher's 'aloev2pm{S}' (the page finds the job by this exact name)", file=sys.stderr)
    tag = "[dry] " if a.dry else ""
    try:
        orig = {k: read(p) for k, p in FILES.items()}
        cur = registered(orig)
        base = orig
        if cur:
            S0, J0 = cur[0]
            if (S0, J0) == (S, JOB):
                print(f"{tag}no-op: p{S} (job name {JOB}) is already registered on the samdata_ft page")
                return 0
            if not a.replace:
                print(f"refused: a different main point is registered: p{S0} (job name {J0}); pass --replace to swap it for p{S} ({JOB})", file=sys.stderr)
                return 2
            base = swap(orig, edits(S0, J0), False, f"unregister p{S0}")
            if registered(base):
                raise Fail(f"unregister p{S0} left a registration behind")
            print(f"{tag}--replace: unregistering p{S0} ({J0})")
        if re.search(rf'Run\("p{S}",', base["builder"]) or re.search(rf"\bp{S}\b", base["template"]):
            raise Fail(f"id p{S} is already used by another run on the page; nothing written")
        new = swap(base, edits(S, JOB), True, f"register p{S}")
        compile(new["builder"], FILES["builder"], "exec")   # syntax check in memory before anything is written
        check_template(new["template"], S, JOB)
    except Fail as ex:
        print(f"FAILED (no change): {ex}", file=sys.stderr)
        return 1
    except SyntaxError as ex:
        print(f"FAILED (no change): the patched builder does not compile: {ex}", file=sys.stderr)
        return 1

    if a.dry:
        out = f"{DRY_ROOT}/{S}"
        os.makedirs(out, exist_ok=True)
        dt = ""
        for k, p in FILES.items():
            dst = f"{out}/{os.path.basename(p)}"
            atomic_write(dst, new[k], os.stat(p).st_mode & 0o777)
            dt += diff_text(orig[k], new[k], p)
        atomic_write(f"{out}/register.diff", dt, 0o644)
        print(dt, end="")
        check_builder(f"{out}/{os.path.basename(FILES['builder'])}")
        print(f"[dry] patched copies in {out}/ (py_compile ok); live files untouched")
        return 0

    baks = {k: f"{p}.bak-monoft-{S}" for k, p in FILES.items()}
    written = []
    try:
        for k, p in FILES.items():
            shutil.copy2(p, baks[k])
            if read(baks[k]) != orig[k]:
                raise Fail(f"backup {baks[k]} does not match {p} (file changed while backing up?)")
        for k, p in FILES.items():   # nobody edited them since we read them
            if read(p) != orig[k]:
                raise Fail(f"{p} changed while registering; aborting")
        for k in ("builder", "template"):
            p = FILES[k]
            atomic_write(p, new[k], os.stat(p).st_mode & 0o777)
            written.append(k)
        check_builder(FILES["builder"])
        check_template(read(FILES["template"]), S, JOB)
        if os.environ.get("SAMDATA_REG_FAILTEST"):   # test hook: exercise the restore path
            raise Fail("injected failure (SAMDATA_REG_FAILTEST)")
        if registered({k: read(p) for k, p in FILES.items()}) != [(S, JOB)]:
            raise Fail(f"post-write check: the registration is not read back as exactly p{S}")
    except BaseException as ex:
        bad = []
        for k in written:
            p = FILES[k]
            try:
                src = read(baks[k])
                if src != orig[k]:
                    src = orig[k]
                atomic_write(p, src, os.stat(p).st_mode & 0o777)
            except BaseException as ex2:
                bad.append(f"{p}: {ex2!r}")
        if bad:
            print(f"FAILED and RESTORE FAILED ({'; '.join(bad)}): restore by hand from {list(baks.values())}. cause: {ex!r}", file=sys.stderr)
        else:
            print(f"FAILED, live files restored from the backups ({', '.join(written) or 'nothing had been written'}): {ex!r}", file=sys.stderr)
        return 1
    print(f"registered p{S} (run dir ppm/aloe-v2pure-main-{S}, job name {JOB}) in {FILES['builder']} + {FILES['template']}; "
          f"backups *.bak-monoft-{S}; the 60 s refresh (job 26082) picks it up on its next build")
    return 0


if __name__ == "__main__":
    sys.exit(main())

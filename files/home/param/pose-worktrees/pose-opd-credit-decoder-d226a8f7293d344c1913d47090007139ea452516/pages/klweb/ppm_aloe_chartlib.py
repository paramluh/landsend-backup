#!/usr/bin/env python3
"""ASR adapter -> PPM live curve page (stdlib only). Rebuilt every 60 s by asr_adapter_refresh.sbatch.

Reads  <run>/metrics.jsonl (+ optional <run>/status.json); run dir = first line of
       /data/home/amartyavarma/ppm/asr-adapter/runs/LIVE unless overridden.
Writes /data/home/amartyavarma/setup-logs/klweb/asr_adapter.html atomically (.tmp + os.replace).

Usage: build_asr_adapter.py [--run RUN_DIR] [--out OUT_HTML] [--overlay OVERLAY_FILE]
       (or env ASR_ADAPTER_RUN / ASR_ADAPTER_OUT / ASR_ADAPTER_OVERLAY; CLI wins).
Overlays (optional file, default runs/OVERLAY): draw each run's train MA + evals in muted colours in the CE/WER/LR
charts, plus a small eval table each. LIVE stays the ★ series; headline + main table are LIVE only."""
import json, os, sys, time, math, html, argparse

LIVE = "/data/home/amartyavarma/ppm/asr-adapter/runs/LIVE"
OVERLAY = "/data/home/amartyavarma/ppm/asr-adapter/runs/OVERLAY"  # optional: one "label|abs run dir" per line
N_OV_COLORS = 3
OUT = "/data/home/amartyavarma/setup-logs/klweb/asr_adapter.html"
REF_TAG, REF_CE, REF_WER = "j19174", 0.4013, 0.321
MA_N = 25
MAX_PTS = 1500  # per drawn train line (decimated; MA computed on full data first)

try:
    from zoneinfo import ZoneInfo
    _PT = ZoneInfo("America/Los_Angeles")
except Exception:
    _PT = None


def pt(ts, fmt="%a %m-%d %H:%M"):
    if ts is None:
        return "—"
    try:
        if _PT is not None:
            import datetime
            return datetime.datetime.fromtimestamp(float(ts), _PT).strftime(fmt) + " PT"
        os.environ["TZ"] = "America/Los_Angeles"; time.tzset()
        return time.strftime(fmt, time.localtime(float(ts))) + " PT"
    except Exception:
        return "—"


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fmt(v, p=4):
    if not num(v):
        return "—"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{p}f}"


def esc(s):
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- inputs
def resolve_run(cli_run):
    if cli_run:
        return cli_run
    if os.environ.get("ASR_ADAPTER_RUN"):
        return os.environ["ASR_ADAPTER_RUN"]
    try:
        with open(LIVE) as f:
            line = f.readline().strip()
        return line or None
    except Exception:
        return None


def load_metrics(run):
    train, evals = {}, {}
    try:
        f = open(os.path.join(run, "metrics.jsonl"), errors="replace")
    except Exception:
        return [], []
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue  # partial last line while trainer writes
            if not isinstance(d, dict) or not num(d.get("step")):
                continue
            k = d.get("kind")
            # keep the last line per step (resumes may rewrite steps)
            if k == "train":
                train[int(d["step"])] = d
            elif k == "eval":
                evals[int(d["step"])] = d
    return [train[s] for s in sorted(train)], [evals[s] for s in sorted(evals)]


def load_overlays(path):
    out = []
    try:
        lines = open(path, errors="replace").read().splitlines()
    except Exception:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        label, d = (x.strip() for x in line.split("|", 1))
        if not d:
            continue
        base = os.path.basename(os.path.normpath(d))
        tr, ev = load_metrics(d) if os.path.isdir(d) else ([], [])
        i = len(out) % N_OV_COLORS + 1
        out.append(dict(label=label or base, dir=d, name=f"{label or base} ({base})", train=tr, evals=ev,
                        exists=os.path.isdir(d), ma=f"ov{i}ma", ln=f"ov{i}ln", mk=f"ov{i}mk"))
    return out


def load_status(run):
    try:
        with open(os.path.join(run, "status.json")) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------- charts
def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t0 = math.floor(lo / step) * step
    t1 = math.ceil(hi / step) * step
    ticks, t = [], t0
    while t <= t1 + step * 1e-9:
        ticks.append(round(t, 12))
        t += step
    return t0, t1, ticks


def log_ticks(lo, hi):
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    ticks = []
    for e in range(a, b + 1):
        for m in (1, 2, 3, 5):
            v = m * 10 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                ticks.append(v)
    if len(ticks) < 2:
        ticks = [lo, hi]
    return ticks


def tick_label(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        m, e = f"{v:.2e}".split("e")
        return f"{m.rstrip('0').rstrip('.')}e{int(e)}"
    if a >= 100:
        return f"{v:.0f}"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s


def decimate(pts, n=MAX_PTS):
    if len(pts) <= n:
        return pts
    stride = len(pts) / n
    out = [pts[int(i * stride)] for i in range(n)]
    if out[-1] is not pts[-1]:
        out.append(pts[-1])
    return out


_CID = [0]


def svg_chart(title, ylabel, series, hlines=(), allow_log=True, xmax_hint=None, height=260):
    """series: list of dict(name, pts=[(x,y)], cls, kind in {line,marker}, tip=fn(x,y)->str).
    hlines: list of (y, label, cls)."""
    W, H = 720, height
    L, R, T, B = 62, 14, 12, 40
    pw, ph = W - L - R, H - T - B
    xs = [x for s in series for x, _ in s["pts"]]
    ys = [y for s in series for _, y in s["pts"]] + [y for y, _, _ in hlines]
    legend = "".join(
        f'<span class="lg"><svg width="22" height="10" aria-hidden="true">'
        + (f'<line x1="1" y1="5" x2="21" y2="5" class="{s["cls"]}"/>' if s["kind"] == "line"
           else f'<circle cx="11" cy="5" r="3.5" class="{s["cls"]}"/>')
        + f'</svg>{esc(s["name"])}</span>' for s in series if not s.get("nolegend"))
    legend += "".join(
        f'<span class="lg"><svg width="22" height="10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" class="{c}"/></svg>{esc(lab)}</span>'
        for _, lab, c in hlines)
    head = f'<figure class="chart"><figcaption>{esc(title)}</figcaption><div class="legend">{legend}</div>'
    if not xs:
        return head + '<div class="empty">no points yet</div></figure>'
    x0, x1 = 0, max(max(xs), xmax_hint or 0, 1)
    _, x1, xt = nice_ticks(x0, x1, 6)
    ymin, ymax = min(ys), max(ys)
    use_log = allow_log and ymin > 0 and ymax / ymin > 10
    if use_log:
        ylo, yhi = ymin / 1.15, ymax * 1.15
        yt = log_ticks(ylo, yhi)
        f = lambda y: T + ph * (1 - (math.log10(y) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)))
    else:
        pad = (ymax - ymin) * 0.08 or (abs(ymax) * 0.1 or 0.01)
        ylo, yhi, yt = nice_ticks(ymin - pad, ymax + pad, 5)
        if ymin >= 0 and ylo < 0:
            ylo = 0.0
            yt = [t for t in yt if t >= 0]
        f = lambda y: T + ph * (1 - (y - ylo) / (yhi - ylo))
    g = lambda x: L + pw * (x - x0) / (x1 - x0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="plot" role="img" aria-label="{esc(title)}">']
    o.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" class="frame"/>')
    for t in yt:
        y = f(t)
        if T - 0.5 <= y <= T + ph + 0.5:
            o.append(f'<line x1="{L}" x2="{L+pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{L-6}" y="{y+4:.1f}" class="tk" text-anchor="end">{tick_label(t)}</text>')
    for t in xt:
        x = g(t)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T+ph}" class="grid"/>'
                 f'<text x="{x:.1f}" y="{T+ph+16}" class="tk" text-anchor="middle">{tick_label(t)}</text>')
    o.append(f'<text x="{L+pw/2:.0f}" y="{H-4}" class="ax" text-anchor="middle">optimizer step</text>')
    o.append(f'<text transform="translate(14 {T+ph/2:.0f}) rotate(-90)" class="ax" text-anchor="middle">'
             f'{esc(ylabel)}{" (log)" if use_log else ""}</text>')
    _CID[0] += 1
    cid = f"clip{_CID[0]}"
    o.append(f'<clipPath id="{cid}"><rect x="{L-5}" y="{T-5}" width="{pw+10}" height="{ph+10}"/></clipPath>')
    o.append(f'<g clip-path="url(#{cid})">')
    for y, lab, c in hlines:
        yy = f(y)
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{yy:.1f}" y2="{yy:.1f}" class="{c}"/>'
                 f'<text x="{L+pw-4}" y="{yy-4:.1f}" class="reflab" text-anchor="end">{esc(lab)}</text>')
    for s in series:
        pts = s["pts"]
        if s["kind"] == "line":
            pts = decimate(pts)
            d = " ".join(f"{g(x):.1f},{f(y):.1f}" for x, y in pts)
            o.append(f'<polyline points="{d}" class="{s["cls"]}"/>')
    for s in series:
        if s["kind"] == "marker":
            tip = s.get("tip")
            for x, y in s["pts"]:
                t = f"<title>{esc(tip(x, y))}</title>" if tip else ""
                o.append(f'<circle cx="{g(x):.1f}" cy="{f(y):.1f}" r="4" class="{s["cls"]}">{t}</circle>')
    o.append("</g></svg>")
    return head + "".join(o) + "</figure>"


def pts_of(rows, key):
    return [(r["step"], r[key]) for r in rows if num(r.get(key))]


def moving_avg(pts, n=MA_N):
    out, acc, q = [], 0.0, []
    for x, y in pts:
        q.append(y); acc += y
        if len(q) > n:
            acc -= q.pop(0)
        out.append((x, acc / len(q)))
    return out


# ---------------------------------------------------------------- page
CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f23;--mut:#6a6f78;--card:#fff;--line:#e3e4e8;--grid:#ececef;
--c1:#2a6fdb;--c2:#d9480f;--c3:#2b8a3e;--ref:#8a8f98;--bad:#c92a2a;--badbg:#fff0f0;--ok:#2b8a3e;
--o1:#9a86b8;--o2:#6fa3a0;--o3:#b89a6a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171b;--fg:#e6e7ea;--mut:#9aa0aa;--card:#1d2026;
--line:#30343c;--grid:#262a31;--c1:#6ea8ff;--c2:#ff8a4c;--c3:#69db7c;--ref:#9aa0aa;--bad:#ff6b6b;--badbg:#3a1d1f;--ok:#69db7c;
--o1:#a596c4;--o2:#7fb3ad;--o3:#c4a77a}}
:root[data-theme="dark"]{--bg:#15171b;--fg:#e6e7ea;--mut:#9aa0aa;--card:#1d2026;--line:#30343c;--grid:#262a31;
--c1:#6ea8ff;--c2:#ff8a4c;--c3:#69db7c;--ref:#9aa0aa;--bad:#ff6b6b;--badbg:#3a1d1f;--ok:#69db7c;
--o1:#a596c4;--o2:#7fb3ad;--o3:#c4a77a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0 6px}
.head{font-size:15px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin:0 0 14px}
.banner{background:var(--badbg);color:var(--bad);border:1px solid var(--bad);border-radius:8px;padding:10px 12px;margin:0 0 12px;font-weight:600}
.warn{color:var(--bad);font-weight:600}
.good{color:var(--ok)}.badv{color:var(--bad)}
.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:0 0 14px;padding:10px 10px 4px}
figcaption{font-weight:600;font-size:14px;margin:0 0 4px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--mut);margin:0 0 4px}
.lg{display:inline-flex;align-items:center;gap:5px}
.plot{width:100%;height:auto;display:block}
.frame{fill:none;stroke:var(--line)}
.grid{stroke:var(--grid);stroke-width:1}
.tk{fill:var(--mut);font-size:11px}.ax{fill:var(--mut);font-size:12px}
.reflab{fill:var(--ref);font-size:11px}
.raw{fill:none;stroke:var(--c1);stroke-width:1;opacity:.3}
.ma{fill:none;stroke:var(--c1);stroke-width:2.2}
.ln2{fill:none;stroke:var(--c2);stroke-width:2}
.ln3{fill:none;stroke:var(--c3);stroke-width:2}
.mk{fill:var(--c2);stroke:var(--card);stroke-width:1.5}
.mk3{fill:var(--c3);stroke:var(--card);stroke-width:1.5}
.ov1ma,.ov1ln{fill:none;stroke:var(--o1);stroke-width:1.8}.ov1mk{fill:var(--o1);stroke:var(--card);stroke-width:1.2}
.ov2ma,.ov2ln{fill:none;stroke:var(--o2);stroke-width:1.8}.ov2mk{fill:var(--o2);stroke:var(--card);stroke-width:1.2}
.ov3ma,.ov3ln{fill:none;stroke:var(--o3);stroke-width:1.8}.ov3mk{fill:var(--o3);stroke:var(--card);stroke-width:1.2}
h2{font-size:14px;margin:16px 0 6px;color:var(--mut)}
.ref{stroke:var(--ref);stroke-width:1.5;stroke-dasharray:6 4;fill:none}
.empty{color:var(--mut);font-size:13px;padding:30px 0;text-align:center}
.two{display:grid;grid-template-columns:1fr 1fr;gap:0 14px}
@media (max-width:700px){.two{grid-template-columns:1fr}}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}th:first-child,td:first-child{text-align:left}
tr:last-child td{border-bottom:0}
.foot{color:var(--mut);font-size:12px;margin:12px 0 30px}
"""


def page(body, title="ASR adapter → PPM"):
    return ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
            "<meta http-equiv=refresh content=60>"
            f"<title>ASR adapter PPM</title><style>{CSS}</style></head><body><main>"
            f"<h1>{esc(title)}</h1>{body}"
            f'<p class="foot">Page rebuilt {esc(pt(time.time(), "%a %m-%d %H:%M:%S"))}; reloads every 60 s.</p>'
            "</main></body></html>")


def delta(v, ref, lower_better=True):
    if not num(v):
        return ""
    d = v - ref
    good = (d < 0) if lower_better else (d > 0)
    return f' <span class="{"good" if good else "badv"}">({"+" if d >= 0 else ""}{d:.4f})</span>'


def eval_row(e):
    fz = e.get("trunk_frozen_ok")
    return "<tr>" + "".join(f"<td>{c}</td>" for c in (
        e["step"], esc(pt(e.get("time"), "%m-%d %H:%M")), fmt(e.get("clip_word_ce_norm")),
        fmt(e.get("clip_word_ce")), fmt(e.get("clip_word_top1"), 3), fmt(e.get("wer"), 3),
        fmt(e.get("text_word_ce_norm")), fmt(e.get("text_kl")),
        "—" if fz is None else ("ok" if fz else '<span class="warn">NO</span>'))) + "</tr>"


def build(run, overlay_path=OVERLAY):
    now = time.time()
    if not run or not os.path.isdir(run):
        msg = "Waiting for the run to start." + ("" if not run else f" (run dir {esc(run)} not found yet)")
        return page(f'<p class="head">{msg}</p><p class="sub">Pointer: {esc(LIVE)}</p>')
    train, evals = load_metrics(run)
    st = load_status(run)
    tag = st.get("run_tag") or os.path.basename(os.path.normpath(run))
    star = f"★ {tag}"
    job = st.get("job") or "—"
    state = st.get("state") or ("running" if train else "starting")
    note = st.get("note") or ""
    parts = []
    if str(state).lower() in ("stopped", "converged", "failed"):
        parts.append(f'<div class="banner">{esc(str(state).upper())}{": " + esc(note) if note else ""}</div>')
    last = train[-1] if train else None
    le = evals[-1] if evals else None
    step = max([r["step"] for r in train] + [r["step"] for r in evals] + [0]) if (train or evals) else None
    h = [f"<b>{esc(star)}</b>", f"job {esc(job)}", f"state <b>{esc(state)}</b>",
         f"step <b>{step if step is not None else '—'}</b>"]
    if le:
        h.append(f"eval norm word CE <b>{fmt(le.get('clip_word_ce_norm'))}</b>{delta(le.get('clip_word_ce_norm'), REF_CE)}"
                 f" vs {REF_TAG} {REF_CE}")
        h.append(f"WER <b>{fmt(le.get('wer'), 3)}</b>{delta(le.get('wer'), REF_WER)} vs {REF_TAG} {REF_WER}")
    else:
        h.append(f"no eval yet (target: {REF_TAG} norm word CE {REF_CE}, WER {REF_WER})")
    parts.append('<p class="head">' + " · ".join(h) + "</p>")
    sub = []
    if last and num(last.get("time")):
        ago = int(now - last["time"])
        s = f"last step: {esc(pt(last['time'], '%a %H:%M:%S'))} ({ago} s ago)"
        if num(last.get("step_s")):
            s += f", {last['step_s']:.2f} s/step"
        sub.append(s)
    elif not train:
        sub.append("no train steps logged yet")
    if num(st.get("started")):
        sub.append(f"started {esc(pt(st['started']))}")
    if note and str(state).lower() not in ("stopped", "converged", "failed"):
        sub.append(esc(note))
    bad_frozen = [e["step"] for e in evals if e.get("trunk_frozen_ok") is False]
    if bad_frozen:
        sub.append(f'<span class="warn">trunk_frozen_ok = false at step {", ".join(map(str, bad_frozen))}</span>')
    parts.append('<p class="sub">' + " · ".join(sub) + "</p>")

    ovs = load_overlays(overlay_path)
    ov_ce, ov_wer, ov_lr = [], [], []
    for ov in ovs:
        n = ov["name"]
        ov_ce.append(dict(name=f"{n} train {MA_N}-step avg", pts=moving_avg(pts_of(ov["train"], "word_ce_norm")),
                          cls=ov["ma"], kind="line"))
        ov_ce.append(dict(name=f"{n} eval", pts=pts_of(ov["evals"], "clip_word_ce_norm"), cls=ov["mk"], kind="marker",
                          tip=lambda x, y, n=n: f"{n} step {x}: eval norm word CE {y:.4f}"))
        w = pts_of(ov["evals"], "wer")
        ov_wer.append(dict(name=n, pts=w, cls=ov["ln"], kind="line"))
        ov_wer.append(dict(name=n, pts=w, cls=ov["mk"], kind="marker", nolegend=True,
                           tip=lambda x, y, n=n: f"{n} step {x}: WER {y:.3f}"))
        ov_lr.append(dict(name=n, pts=pts_of(ov["train"], "lr"), cls=ov["ln"], kind="line"))
    xmax = step or 0
    ce_raw = pts_of(train, "word_ce_norm")
    ce_eval = pts_of(evals, "clip_word_ce_norm")
    parts.append(svg_chart(
        f"Transcript normalised word CE · {star}", "norm word CE",
        ov_ce + [dict(name=f"{star} train", pts=ce_raw, cls="raw", kind="line"),
         dict(name=f"{star} train {MA_N}-step avg", pts=moving_avg(ce_raw), cls="ma", kind="line"),
         dict(name=f"{star} eval", pts=ce_eval, cls="mk", kind="marker",
              tip=lambda x, y: f"step {x}: eval norm word CE {y:.4f}")],
        hlines=[(REF_CE, f"{REF_TAG} final {REF_CE}", "ref")], xmax_hint=xmax))
    parts.append(svg_chart(
        f"Eval WER · {star}", "WER",
        ov_wer + [dict(name=f"{star} eval WER", pts=pts_of(evals, "wer"), cls="ln2", kind="line"),
         dict(name=f"{star} eval WER (points)", pts=pts_of(evals, "wer"), cls="mk", kind="marker", nolegend=True,
              tip=lambda x, y: f"step {x}: WER {y:.3f}")],
        hlines=[(REF_WER, f"{REF_TAG} final {REF_WER}", "ref")], allow_log=False, xmax_hint=xmax))
    parts.append(svg_chart(
        f"LR · {star}", "learning rate",
        ov_lr + [dict(name=f"{star} lr", pts=pts_of(train, "lr"), cls="ma", kind="line")],
        allow_log=False, xmax_hint=xmax, height=200))
    tce = pts_of(evals, "text_word_ce_norm")
    tkl = pts_of(evals, "text_kl")
    parts.append('<div class="two">' + svg_chart(
        f"Text-duplex held-out norm word CE (should be flat) · {star}", "text norm word CE",
        [dict(name=f"{star} text CE", pts=tce, cls="ln3", kind="line"),
         dict(name=f"{star} text CE (points)", pts=tce, cls="mk3", kind="marker", nolegend=True,
              tip=lambda x, y: f"step {x}: text norm word CE {y:.4f}")],
        allow_log=False, xmax_hint=xmax, height=220) + svg_chart(
        f"Text-duplex KL vs frozen PPM (should be ~0) · {star}", "text KL",
        [dict(name=f"{star} text KL", pts=tkl, cls="ln3", kind="line"),
         dict(name=f"{star} text KL (points)", pts=tkl, cls="mk3", kind="marker", nolegend=True,
              tip=lambda x, y: f"step {x}: text KL {y:.3g}")],
        allow_log=False, xmax_hint=xmax, height=220) + "</div>")

    rows = [eval_row(e) for e in reversed(evals)]
    rows.append(f"<tr><td>{REF_TAG} final (ref)</td><td>—</td><td>{REF_CE}</td><td>—</td><td>—</td>"
                f"<td>{REF_WER}</td><td>—</td><td>—</td><td>—</td></tr>")
    parts.append('<div class="tw"><table><thead><tr><th>step</th><th>time (PT)</th><th>eval norm CE</th><th>full CE</th>'
                 '<th>top-1</th><th>WER</th><th>text CE</th><th>text KL</th><th>trunk frozen</th></tr></thead><tbody>'
                 + "".join(rows) + "</tbody></table></div>")
    for ov in ovs:
        parts.append(f'<h2>Overlay: {esc(ov["name"])}</h2>')
        if not ov["evals"]:
            parts.append(f'<p class="sub">{"no evals yet" if ov["exists"] else "run dir not found"}: {esc(ov["dir"])}</p>')
            continue
        parts.append('<div class="tw"><table><thead><tr><th>step</th><th>time (PT)</th><th>eval norm CE</th><th>full CE</th>'
                     '<th>top-1</th><th>WER</th><th>text CE</th><th>text KL</th><th>trunk frozen</th></tr></thead><tbody>'
                     + "".join(eval_row(e) for e in reversed(ov["evals"])) + "</tbody></table></div>")
    parts.append(f'<p class="foot">Run dir {esc(run)} · {len(train)} train lines, {len(evals)} evals'
                 + "".join(f" · overlay {esc(ov['dir'])}: {len(ov['train'])} train, {len(ov['evals'])} evals" for ov in ovs)
                 + ".</p>")
    return page("".join(parts)), (tag, state, step, len(train), len(evals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=os.environ.get("ASR_ADAPTER_OUT", OUT))
    ap.add_argument("--overlay", default=os.environ.get("ASR_ADAPTER_OVERLAY", OVERLAY))
    a = ap.parse_args()
    run = resolve_run(a.run)
    try:
        res = build(run, a.overlay)
    except Exception as ex:  # never leave the page stale silently
        import traceback; traceback.print_exc()
        res = page(f'<div class="banner">page builder error: {esc(repr(ex))}</div><p class="sub">run {esc(run)}</p>')
    info = None
    if isinstance(res, tuple):
        res, info = res
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write(res)
    os.replace(tmp, a.out)
    print("asr_adapter.html", "waiting" if info is None else
          "★ %s state %s step %s train %d eval %d" % info, "->", a.out)


if __name__ == "__main__":
    main()

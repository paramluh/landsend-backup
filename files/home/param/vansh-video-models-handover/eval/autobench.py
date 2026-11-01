"""
AutoBench (docs/corruption/autobench.md): drift of long autoregressive talking-person videos. Every 1 fps frame
(pixel frame 25 t) is scored against the same video's reference frame: the sink (t = 0), or t = --ref s. A sample dir
holds sample_<n>.mp4 (the generation) and samples_<n>.mp4 (GT, scored the same way as the natural-drift baseline);
with --sxs BAND, labelled side-by-sides NN_*.mp4 instead (BAND label rows on top, GT left half, generation right half).

Regions: BiRefNet-portrait person probability > 0.5 (input 1024 x 1024); person and background each eroded by 7 px and
intersected with the reference frame's, so both frames are compared over the same pixels. Lab = CIE L*a*b* (D65) of
the decoded sRGB; E = region mean of sum_c (Lab - G_1.5 * Lab)^2.
  m1               background CIE76 ΔE between the region-mean Lab of frame t and of the reference   (AutoBench M1)
  m2               background ln(E_t / E_ref)                                                         (AutoBench M2)
  person_de        person CIE76 ΔE between the region-mean Lab of frame t and of the reference (M1 on the person)
  person_de_vs_gt  gen only: |Δ_gen - Δ_GT|, Δ = that person region-mean Lab change of each video at the same
                   second; gen and GT share sink, pose and lighting, so most of the person's motion cancels
  person_tex       person ln(E_t / E_ref)
  identity         ArcFace cosine (InsightFace buffalo_l, largest face) vs the reference's face; NaN without a face
Per video: minute means (minute 1 from t = ref + 1 s); over = m1 > 1.0 for >= 5 s in a row (onset = its first
second); seconds over 1.0; end ΔE = mean m1 over 293-297 s; mosaic = minutes 2-5 with mean m1 > 2 and mean m2 > 1.

Usage:
    .venv/bin/python eval/autobench.py DIR [DIR ...] --out OUT [--gpus 0,1,...] [--ref 1] [--clips 0,3]
        [--seconds 0:300:3,293:298] [--sxs 128] [--plots]
DIRs (globs allowed) are searched recursively for sample dirs. Output OUT/<run>/: clip_<n>.csv (per second, gen and
GT columns), clips.csv (derived scores per clip and video), summary.md, plots/clip_<n>.png; OUT/summary.{md,csv}: one
row per run. Clips with a CSV are not rescored.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.environ.get('ARBENCH_PKGS', '/tmp/arbench_pkgs'))

SEG, SEG_REVISION, SEG_RES = 'ZhengPeng7/BiRefNet-portrait', 'b6561965a70070d9143fd9e558f6ca3c481510db', 1024
ERODE, SIGMA = 7, 1.5
OVER, RUN, END, MINUTES, MOSAIC = 1.0, 5, (293, 297), 5, (2.0, 1.0)
METRICS = ['m1', 'm2', 'person_de', 'person_de_vs_gt', 'person_tex', 'identity']
VIDEOS = ['gen', 'gt']


def read_frames(path, threads):
    """-> uint8 [T, H, W, 3] at pixel frames 0, fps, 2 fps, ... and their picture types. RGB is ffmpeg's default
    conversion of untagged yuv420p (BT.601, limited range), the inverse of the encoder's (inference/common.py)."""
    w, h, rate = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', path],
        capture_output=True, text=True, check=True,
    ).stdout.split()[0].split(',')
    num, den = map(int, rate.split('/'))
    assert num % den == 0, rate
    r = subprocess.run(
        ['ffmpeg', '-nostats', '-threads', str(threads), '-i', path, '-vf', f"select='not(mod(n\\,{num // den}))',showinfo",
         '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        capture_output=True, check=True,
    )
    types = re.findall(r'iskey:\d type:(\w)', r.stderr.decode())
    frames = np.frombuffer(bytearray(r.stdout), np.uint8).reshape(-1, int(h), int(w), 3)
    assert len(types) == len(frames), (len(types), len(frames))
    return frames, types


class Scorer:
    def __init__(self, device, threads):
        import kornia
        import onnxruntime
        from insightface.app import FaceAnalysis
        from transformers import AutoModelForImageSegmentation

        cuda = device == 'cuda'
        self.device, self.dtype, self.batch = device, torch.float16 if cuda else torch.float32, 16 if cuda else 1
        self.seg = AutoModelForImageSegmentation.from_pretrained(SEG, revision=SEG_REVISION, trust_remote_code=True)
        self.seg = self.seg.to(device, self.dtype).eval()
        self.lab, self.blur = kornia.color.rgb_to_lab, kornia.filters.gaussian_blur2d
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        r = torch.arange(-ERODE, ERODE + 1)
        self.disk = ((r[:, None] ** 2 + r ** 2) <= ERODE ** 2).float().to(device)[None, None]
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.face = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=providers,
                                 sess_options=options)
        self.face.prepare(ctx_id=0 if cuda else -1, det_size=(640, 640))
        print(f'[autobench] {device}: identity on {self.face.models["recognition"].session.get_providers()[0]}', flush=True)

    def erode(self, m):
        return F.conv2d((~m).float()[:, None], self.disk, padding=ERODE)[:, 0] < 0.5

    def person(self, x):
        """RGB [B, 3, H, W] in [0, 1] -> person mask [B, H, W] bool."""
        y = F.interpolate(x, (SEG_RES, SEG_RES), mode='bilinear', align_corners=False)
        p = self.seg(((y - self.mean) / self.std).to(self.dtype))[-1].float().sigmoid()
        return F.interpolate(p, x.shape[2:], mode='bilinear', align_corners=False)[:, 0] > 0.5

    @torch.no_grad()
    def regions(self, frames):
        """uint8 [B, H, W, 3] -> eroded person, eroded background [B, H, W] bool, Lab [B, 3, H, W], E map [B, H, W]."""
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255
        m = self.person(x)
        lab = self.lab(x)
        e = ((lab - self.blur(lab, (11, 11), (SIGMA, SIGMA), border_type='reflect')) ** 2).sum(1)
        return self.erode(m), self.erode(~m), lab, e

    def embed(self, frame):
        faces = self.face.get(np.ascontiguousarray(frame[:, :, ::-1]))
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding if faces else None

    @torch.no_grad()
    def video(self, frames, ref):
        """frames uint8 [T, H, W, 3], ref = the reference frame's index -> {column: [T]}, person Lab change [T, 3]."""
        P0, B0, lab0, e0 = (v[0] for v in self.regions(frames[ref:ref + 1]))
        out = {k: [] for k in ['m1', 'm2', 'person_de', 'person_tex', 'n_bg', 'n_person']}
        change = []
        for i in range(0, len(frames), self.batch):
            for P, B, lab, e in zip(*self.regions(frames[i:i + self.batch])):
                bg, person = B & B0, P & P0
                out['m1'].append(float((lab[:, bg].mean(1) - lab0[:, bg].mean(1)).norm()))
                out['m2'].append(float(torch.log(e[bg].mean() / e0[bg].mean())))
                change.append(lab[:, person].mean(1) - lab0[:, person].mean(1))
                out['person_tex'].append(float(torch.log(e[person].mean() / e0[person].mean())))
                out['n_bg'].append(int(bg.sum()))
                out['n_person'].append(int(person.sum()))
        change = torch.stack(change)
        out['person_de'] = change.norm(dim=1).tolist()
        faces = [self.embed(f) for f in frames]
        out['identity'] = [float(f @ faces[ref]) if f is not None and faces[ref] is not None else float('nan') for f in faces]
        return out, change


def parse_seconds(spec, T, ref):
    if spec is None:
        return list(range(T))
    keep = {ref}
    for part in spec.split(','):
        keep |= set(range(T)[slice(*[int(x) if x else None for x in part.split(':')])])
    return sorted(keep)


def score_clip(scorer, run, files, args, path):
    t0 = time.time()
    if args.sxs is None:
        videos = {v: read_frames(os.path.join(run, f), args.threads) for v, f in zip(VIDEOS, files)}
    else:
        frames, types = read_frames(os.path.join(run, files[0]), args.threads)
        W = frames.shape[2] // 2
        videos = {'gen': (frames[:, args.sxs:, W:], types), 'gt': (frames[:, args.sxs:, :W], types)}
    seconds = parse_seconds(args.seconds, min(len(f) for f, _ in videos.values()), args.ref)
    cols, change = {'t': seconds}, {}
    for v, (frames, types) in videos.items():
        res, change[v] = scorer.video(frames[seconds], seconds.index(args.ref))
        cols[f'{v}_type'] = [types[t] for t in seconds]
        cols.update({f'{v}_{k}': x for k, x in res.items()})
    cols['gen_person_de_vs_gt'] = (change['gen'] - change['gt']).norm(dim=1).tolist()
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(zip(*[[f'{x:.6g}' if isinstance(x, float) else x for x in c] for c in cols.values()]))
    print(f'[autobench] {path}: {len(seconds)} s in {time.time() - t0:.0f} s', flush=True)


def nanmean(v, f=np.nanmean):
    v = np.asarray(v, float)
    return float(f(v)) if np.isfinite(v).any() else float('nan')


def load_clip(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([int(r['t']) for r in rows])
    return t, {v: {k[len(v) + 1:]: np.array([float(r[k]) for r in rows]) for k in rows[0] if k.startswith(f'{v}_') and k != f'{v}_type'}
               for v in VIDEOS}


def derived(t, c, ref):
    """t [T] seconds, c {column: [T]} of one video -> its scores."""
    after = t > ref
    blocks = {k: [nanmean(c[k][after & (t >= 60 * i) & (t < 60 * (i + 1))]) if k in c else float('nan') for i in range(MINUTES)]
              for k in METRICS}
    above = (c['m1'] > OVER) & after
    onset = next((int(t[i]) for i in range(len(t) - RUN + 1) if above[i:i + RUN].all() and t[i + RUN - 1] - t[i] == RUN - 1), None)
    d = {
        'over': int(onset is not None), 'onset': onset, 'seconds_over': int(above.sum()),
        'end_de': nanmean(c['m1'][(t >= END[0]) & (t <= END[1])]),
        'mosaic': ' '.join(str(i + 1) for i in range(1, MINUTES) if blocks['m1'][i] > MOSAIC[0] and blocks['m2'][i] > MOSAIC[1]),
        'no_face_s': int(np.isnan(c['identity'][after]).sum()),
        'min_bg_frac': float((c['n_bg'] / c['n_bg'][t == ref]).min()),
        'min_person_frac': float((c['n_person'] / c['n_person'][t == ref]).min()),
    }
    d.update({f'{k}_min{i + 1}': blocks[k][i] for k in METRICS for i in range(MINUTES)})
    return d


def fmt(v, p=2, missing='–'):
    return missing if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{p}f}' if isinstance(v, float) else str(v)


def summarise(run, label, clips, args):
    """clip CSVs of one run -> clips.csv, summary.md, plots; -> the run's summary row."""
    out = os.path.join(args.out, label)
    data = {n: load_clip(os.path.join(out, f'clip_{n}.csv')) for n in clips}
    D = {(n, v): derived(t, c[v], args.ref) for n, (t, c) in data.items() for v in VIDEOS}
    with open(os.path.join(out, 'clips.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        keys = list(D[next(iter(D))])
        w.writerow(['clip', 'video'] + keys)
        w.writerows([n, v] + [fmt(d[k], 4, '') for k in keys] for (n, v), d in D.items())

    seconds = sum(int((t > args.ref).sum()) for t, _ in data.values())
    S = {}
    for v in VIDEOS:
        d = [D[n, v] for n in clips]
        S[v] = {
            'clips over': f'{sum(x["over"] for x in d)} / {len(d)}',
            'seconds over 1.0': f'{sum(x["seconds_over"] for x in d)} / {seconds}',
            'minute-5 ΔE': fmt(nanmean([x['m1_min5'] for x in d])),
            'end ΔE mean (median)': f'{fmt(nanmean([x["end_de"] for x in d]))} ({fmt(nanmean([x["end_de"] for x in d], np.nanmedian))})',
            'mosaic clips': str(sum(bool(x['mosaic']) for x in d)),
            'M2 minute 5': fmt(nanmean([x['m2_min5'] for x in d])),
            'person ΔE minute 5': fmt(nanmean([x['person_de_min5'] for x in d])),
            'person ΔE vs GT minute 5': fmt(nanmean([x['person_de_vs_gt_min5'] for x in d])),
            'person texture minute 5': fmt(nanmean([x['person_tex_min5'] for x in d])),
            'identity minute 5': fmt(nanmean([x['identity_min5'] for x in d])),
            'seconds without a face': str(sum(x['no_face_s'] for x in d)),
        }
    gt_m2 = nanmean([c['gt']['m2'][t > args.ref] for t, c in data.values()])
    L = [f'# AutoBench: {label}', '', f'`{run}`, {len(clips)} clips, reference t = {args.ref} s'
         + (' (the sink)' if args.ref == 0 else '') + (f', seconds {args.seconds}' if args.seconds else '') + '.',
         f'GT M2 mean {gt_m2:.2f} (real studio footage in full-quality files: +0.2 to +0.3; compression raises it).', '',
         '| score | gen | GT |', '|---|---:|---:|']
    L += [f'| {k} | {S["gen"][k]} | {S["gt"][k]} |' for k in S['gen']]
    L += ['', 'Minute means over clips, gen / GT:', '', '| metric | ' + ' | '.join(f'min {i + 1}' for i in range(MINUTES)) + ' |',
          '|---|' + '---:|' * MINUTES]
    for k in METRICS:
        L.append(f'| {k} | ' + ' | '.join(' / '.join(fmt(nanmean([D[n, v][f"{k}_min{i + 1}"] for n in clips])) for v in VIDEOS)
                                          for i in range(MINUTES)) + ' |')
    L += ['', 'Per clip, gen / GT:', '', '| clip | onset | s over | end ΔE | min-5 ΔE | mosaic | M2 min 5 | person ΔE min 5 | '
          'gen person ΔE vs GT min 5 | person tex min 5 | identity min 5 |', '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for n in clips:
        g, r = D[n, 'gen'], D[n, 'gt']
        pair = lambda k: f'{fmt(g[k])} / {fmt(r[k])}'
        L.append(f'| {n} | {fmt(g["onset"])} / {fmt(r["onset"])} | {g["seconds_over"]} / {r["seconds_over"]} | {pair("end_de")} | '
                 f'{pair("m1_min5")} | {g["mosaic"] or "–"} / {r["mosaic"] or "–"} | {pair("m2_min5")} | {pair("person_de_min5")} | '
                 f'{fmt(g["person_de_vs_gt_min5"])} | {pair("person_tex_min5")} | {pair("identity_min5")} |')
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    if args.plots:
        plot(out, data)
    return S


def plot(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    lines = {'m1': [OVER, MOSAIC[0]], 'm2': [0, MOSAIC[1]], 'person_de': [OVER], 'person_de_vs_gt': [OVER], 'person_tex': [0],
             'identity': [0.8]}
    for n, (t, c) in data.items():
        fig, axs = plt.subplots(1, len(METRICS), figsize=(4.5 * len(METRICS), 3.2))
        for ax, k in zip(axs, METRICS):
            for y in lines[k]:
                ax.axhline(y, color='k', ls=':', lw=0.8)
            if k in c['gt']:
                ax.plot(t, c['gt'][k], color='0.45', lw=1, label='GT')
            ax.plot(t, c['gen'][k], color='tab:red', lw=1, label='gen')
            ax.set_title(k)
            ax.set_xlabel('t (s)')
            ax.grid(alpha=0.3)
        axs[0].legend()
        fig.suptitle(f'{out} clip {n}')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'plots', f'clip_{n}.png'), dpi=80)
        plt.close(fig)


def find_runs(patterns, sxs):
    """-> {dir: [(clip, (gen file, GT file) or (side-by-side file,))]} for every dir under the patterns holding clips."""
    runs = {}
    for root in sorted({p for g in patterns for p in glob.glob(os.path.expanduser(g))}):
        for d, _, files in os.walk(root, followlinks=True):
            if sxs is None:
                clips = {int(m[1]): (f, f'samples_{m[1]}.mp4') for f in files if (m := re.fullmatch(r'sample_(\d+)\.mp4', f))}
                missing = [g for _, g in clips.values() if g not in files]
                assert not missing, f'{d}: no GT {missing}'
            else:
                clips = {int(m[1]): (f,) for f in files if (m := re.fullmatch(r'(\d+)_.*\.mp4', f))}
            if clips:
                runs[os.path.abspath(d)] = sorted(clips.items())
    return dict(sorted(runs.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+', help='sample dirs, or dirs searched for them (globs allowed)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref', type=int, default=0, help='reference second (0 = the sink)')
    ap.add_argument('--seconds', help='score only these seconds, e.g. 0:300:3,293:298 (the reference is always scored)')
    ap.add_argument('--clips', help='only these clip ids, e.g. 0,3')
    ap.add_argument('--sxs', type=int, metavar='BAND', help='labelled side-by-sides NN_*.mp4 with BAND label rows')
    ap.add_argument('--gpus', help='one worker per GPU, e.g. 0,1,2,3; without it the clips are scored on the CPU')
    ap.add_argument('--threads', type=int, default=4, help='CPU threads for torch, onnxruntime and ffmpeg')
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    runs = find_runs(args.dirs, args.sxs)
    if args.clips:
        keep = {int(x) for x in args.clips.split(',')}
        runs = {d: [c for c in clips if c[0] in keep] for d, clips in runs.items() if any(c[0] in keep for c in clips)}
    assert runs, f'no clips under {args.dirs}'
    base = os.path.dirname(next(iter(runs))) if len(runs) == 1 else os.path.commonpath(list(runs))
    labels = {d: os.path.relpath(d, base) for d in runs}
    jobs = [(d, files, os.path.join(args.out, labels[d], f'clip_{n}.csv')) for d, clips in runs.items() for n, files in clips]
    if args.worker is not None:
        jobs = jobs[args.worker::len(args.gpus.split(','))]
    todo = [j for j in jobs if not os.path.exists(j[2])]
    if args.gpus is not None and args.worker is None:
        procs = [subprocess.Popen([sys.executable, *sys.argv, '--worker', str(k)], env={**os.environ, 'CUDA_VISIBLE_DEVICES': g})
                 for k, g in enumerate(args.gpus.split(','))]
        assert not any(p.wait() for p in procs), 'a worker failed'
    elif todo:
        torch.set_num_threads(args.threads)
        scorer = Scorer('cpu' if args.worker is None else 'cuda', args.threads)
        for d, files, path in todo:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            score_clip(scorer, d, files, args, path)
    if args.worker is not None:
        return

    rows = [(labels[d], len(clips), summarise(d, labels[d], [n for n, _ in clips], args)) for d, clips in runs.items()]
    keys = ['clips over', 'seconds over 1.0', 'minute-5 ΔE', 'end ΔE mean (median)', 'mosaic clips', 'M2 minute 5',
            'person ΔE minute 5', 'person ΔE vs GT minute 5', 'identity minute 5']
    L = [f'# AutoBench, reference t = {args.ref} s' + (f', seconds {args.seconds}' if args.seconds else ''), '',
         '| run | video | clips | ' + ' | '.join(keys) + ' |', '|---|---|---:|' + '---:|' * len(keys)]
    L += [f'| {label} | {v} | {n} | ' + ' | '.join(S[v][k] for k in keys) + ' |' for label, n, S in rows for v in VIDEOS]
    open(os.path.join(args.out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'clips', 'video'] + list(rows[0][2]['gen']))
        w.writerows([label, n, v] + list(S[v].values()) for label, n, S in rows for v in VIDEOS)
    print(open(os.path.join(args.out, 'summary.md')).read())


if __name__ == '__main__':
    main()

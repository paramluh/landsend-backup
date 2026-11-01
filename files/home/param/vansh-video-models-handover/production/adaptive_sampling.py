"""
Adaptive sampling: production/cached_universal.py's sampler for one clip, plus an online background-colour drift
detector (streaming causal VAE decode with its persistent cache; ARBench background ΔE of the newest pixel frame vs the
sink, checked once per generated second, its decision held until the next check). While the last check is over
--threshold, the history is corrupted with the [history_corruption] block of --toml, every active step (color_p /
mosaic_p rescaled to sum to 1; `slots` = full only):
  rewrite: at the start of active step k, a fresh draw on the clean history k-H..k-1 (all real slots) is committed into
           window k's entries only (each (frame, slot) entry is read by one window). The generated latents and every
           other entry stay clean: windows are 1111 while active, 0000 right after.
  commit:  corrupt-and-cache. At an onset step one draw (Episode) is held for the episode; every active step k commits
           frame k-1 once more as x + d into all H of its entries (none read yet, so the same as committing it
           corrupted once). Nothing already read is rewritten: windows go 0001 -> 0011 -> 0111 -> 1111 and after the
           episode 1110 -> 1100 -> 1000. The generated latents stay clean.

Usage:
    python adaptive_sampling.py CONFIG --mode rewrite|commit --clip N --out DIR [--toml TOML] [--threshold T]
        [--stop-frame F] [--no-corrupt] [--manifest CSV]
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(os.environ.get('REPO', Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO / 'production'))
sys.path.insert(0, str(REPO))
sys.path.append(os.environ.get('ARBENCH_PKGS', '/tmp/arbench_pkgs'))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from torch.cuda import amp

from cached_universal import SCRATCH, CachedNet
from inference.common import FPS, decode_wan22_vae, load_sample
from inference.loader import build_models
from models.autoencoders.wanvae import unpatchify
from models.corruption import HistoryCorruption, _log_uniform, background_mask, pose_background_mask
from models.mosaic import mosaic_pattern

ROOT = '/mnt/paramfs/vansh/aloe-conversational/conv-4.3k-5mins-muon-neutrino-256x480-v1'
TOML = REPO / 'experiments/universal/aloe-h4-complete-corruption.toml'
SEG, SEG_RES, ERODE, SIGMA = 'ZhengPeng7/BiRefNet-portrait', 1024, 7, 1.5


def build_sample(samples_dir, manifest, n):
    if (samples_dir / f'samples_{n}_low_res.safetensors').exists():
        return
    row = next(r for r in csv.DictReader(open(manifest)) if int(r['index']) == n)
    d = os.path.join(ROOT, row['paths'])
    samples_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(f'{d}/video_latents.safetensors', 'pt') as h:
        latents = h.get_tensor('latents').permute(1, 0, 2, 3).contiguous()
    with safe_open(f'{d}/pose_latents.safetensors', 'pt') as h:
        pose = h.get_tensor('latents').permute(1, 0, 2, 3).contiguous()
    with safe_open(f'{d}/mimi.safetensors', 'pt') as h:
        codes = h.get_tensor('codes').int().numpy()
    save_file({'latents': latents}, str(samples_dir / f'samples_{n}_low_res.safetensors'))
    save_file({'pose_latents': pose}, str(samples_dir / f'samples_{n}_pose.safetensors'))
    np.save(samples_dir / f'samples_{n}_mimi.npy', codes)


def corruption_config(path):
    cfg = tomllib.load(open(path, 'rb'))['history_corruption']
    total = cfg['color_p'] + cfg['mosaic_p']
    return cfg | {'color_p': cfg['color_p'] / total, 'mosaic_p': cfg['mosaic_p'] / total, 'slots': [1.0, 0.0, 0.0]}


class Episode:
    """Commit-mode d (models/corruption.py terms, one window's draw): drawn at the onset step with the scale of the onset
    window's clean history [C, S, h, w]; mosaic (probability mosaic_p) or colour. Frame f is committed as
    x_f + scale * d_f, d_f's wall mask / mosaic envelope from f's own clean latent and pose. The mosaic's random fields
    come from one seed, so they are the same for every frame."""

    def __init__(self, c, history):
        C = history.shape[0]
        device = history.device
        self.c = c
        self.scale = history.flatten(1).std(1, correction=0)[:, None, None]
        self.mosaic = float(torch.rand(1)) < c.mosaic_p
        choice = int(torch.multinomial(c.directions, 1))
        direction = c.measured[choice].to(device) if choice < 2 else torch.randn(C, device=device)
        sign = int(torch.randint(0, 2, (1,))) * 2 - 1
        self.color = (sign * math.sqrt(C) * direction / direction.norm())[:, None, None]
        self.a_person = float(_log_uniform(c.person_color, 1, device))
        self.a_background = float(_log_uniform(c.background_color, 1, device))
        self.info = {'direction': ['dark', 'drift', 'random'][choice], 'sign': sign, 'a_person': round(self.a_person, 4),
                     'a_background': round(self.a_background, 4)}
        if self.mosaic:
            self.a = _log_uniform(c.mosaic, 1, device)
            self.coverage = torch.empty(1, device=device).uniform_(*c.mosaic_coverage)
            self.seed = int(torch.randint(2 ** 62, (1,)))
            self.info = {'mosaic': round(float(self.a), 4), 'coverage': round(float(self.coverage), 4)}

    def __call__(self, latent, pose):
        c = self.c
        if self.mosaic:
            with torch.random.fork_rng([latent.device]):
                torch.cuda.manual_seed(self.seed)
                envelope = pose_background_mask(pose[None], c.mask_tau_pose)
                return latent + self.scale * mosaic_pattern(envelope, self.a, c.mosaic_dc * self.a, self.coverage)[0]
        m = background_mask(latent[None], pose[None], c.mask_tau, c.mask_tau_pose)[0]
        return latent + self.scale * (self.color * (self.a_person + m * self.a_background))


class Stream:
    """Wan2_2_VAE.decode one latent frame at a time with the persistent causal cache -> uint8 [T, H, W, 3]."""

    def __init__(self, vae):
        self.vae, self.model = vae, vae.model
        self.model.clear_cache()
        self.i = 0

    @torch.no_grad()
    def __call__(self, z):
        model, (mean, inv_std) = self.model, self.vae.scale
        with amp.autocast(dtype=self.vae.dtype):
            z = z[None, :, None].to(self.vae.device, self.vae.dtype)
            z = z / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
            model._conv_idx = [0]
            out = model.decoder(model.conv2(z), feat_cache=model._feat_map, feat_idx=model._conv_idx,
                                first_chunk=self.i == 0)
            out = unpatchify(out, patch_size=2).float().clamp_(-1, 1)
        self.i += 1
        return out[0].clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 3, 0)


class Detector:
    """ARBench color_bg, color_person, tex_bg of one frame vs the sink (BiRefNet-portrait > 0.5, 7 px erosion)."""

    def __init__(self):
        from transformers import AutoModelForImageSegmentation
        import kornia
        self.lab, self.blur = kornia.color.rgb_to_lab, kornia.filters.gaussian_blur2d
        self.seg = AutoModelForImageSegmentation.from_pretrained(SEG, trust_remote_code=True).cuda().eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
        yy, xx = torch.meshgrid(torch.arange(-ERODE, ERODE + 1), torch.arange(-ERODE, ERODE + 1), indexing='ij')
        self.disk = ((yy ** 2 + xx ** 2) <= ERODE ** 2).float().cuda()[None, None]

    @torch.no_grad()
    def regions(self, frame):
        x = frame.permute(2, 0, 1)[None].float() / 255
        H, W = x.shape[2:]
        y = F.interpolate(x, (SEG_RES, SEG_RES), mode='bilinear', align_corners=False)
        p = self.seg(((y - self.mean) / self.std).half())[-1].float().sigmoid()
        m = F.interpolate(p, (H, W), mode='bilinear', align_corners=False)[:, 0] > 0.5
        background = F.conv2d(m.float()[:, None], self.disk, padding=ERODE)[:, 0] < 0.5
        person = F.conv2d((~m).float()[:, None], self.disk, padding=ERODE)[:, 0] < 0.5
        lab = self.lab(x)
        e = ((lab - self.blur(lab, (11, 11), (SIGMA, SIGMA), border_type='reflect')) ** 2).sum(1)
        return background, person, lab, e

    def set_sink(self, frame):
        self.b0, self.p0, self.lab0, self.e0 = self.regions(frame)

    @torch.no_grad()
    def __call__(self, frame):
        """-> (background dE, person dE, background texture) vs the sink."""
        b, p, lab, e = self.regions(frame)
        M = (b & self.b0).float()
        mean = lambda L: (L * M[:, None]).sum((-2, -1)) / M.sum((-2, -1))[:, None]
        bg = float((mean(lab) - mean(self.lab0)).norm(dim=1)[0])
        pmean = lambda L, P: (L * P[:, None]).sum((-2, -1)) / P.sum((-2, -1))[:, None]
        person = float((pmean(lab, p.float()) - pmean(self.lab0, self.p0.float())).norm(dim=1)[0])
        tex = float(torch.log((e * M).sum() / M.sum()) - torch.log((self.e0 * M).sum() / M.sum()))
        return bg, person, tex


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--mode', required=True, choices=['rewrite', 'commit'])
    ap.add_argument('--clip', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--toml', default=str(TOML), help='training toml whose [history_corruption] block is used')
    ap.add_argument('--threshold', type=float, default=1.0, help='background colour dE')
    ap.add_argument('--stop-frame', type=int, default=None, help='stop after this many latent frames, latents only')
    ap.add_argument('--no-corrupt', action='store_true', help='detect and log only')
    ap.add_argument('--manifest', default=None, help='clip manifest (index, paths) to build missing samples')
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(open(config_path))
    assert config['audio_guidance_scale'] == 1.0 or config['audio_guidance_steps'] == 0
    samples_dir = Path(config['samples_dir'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, device = args.clip, 'cuda'
    if args.manifest:
        build_sample(samples_dir, args.manifest, n)
    ccfg = corruption_config(args.toml)
    corrupt = HistoryCorruption(ccfg)
    name, tag = f'sample_{n}', f'{args.mode}:{n}'

    models = build_models(config, config_path.parent, device)
    latents, pose, code_table, codes = load_sample(samples_dir, n, models.num_codebooks, config['max_latent_frames'],
                                                   slide_audio=True)
    cached = CachedNet(models, latents.shape[2:], config['substeps'], device, config['compile'], 1.0)
    assert cached.buf_hidden is None, 'adaptive sampling never sets hiddens; text models are unsupported'
    H, P, m = cached.H, cached.P, cached.m
    C, L, h, w = latents.shape
    latents, pose = latents.to(device), pose.to(device)
    code_table, codes = code_table.to(device), codes.to(device)
    seed = config['seed'] + n
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(C, L, h, w, generator=generator, device=device)
    cached.start_clip(latents[None, :, 0], config['cuda_graph'])
    rows, commits = cached.schedule(L)
    audio_frames = cached.audio_schedule(L, L)
    end = L if args.stop_frame is None else min(L, args.stop_frame)

    generated = torch.zeros(C, L, h, w, device=device)
    generated[:, 0] = latents[:, 0]
    stream, detector = Stream(models.vae), Detector()
    torch.manual_seed(seed)
    log = {'clip': n, 'seed': seed, 'mode': args.mode, 'threshold': args.threshold, 'corrupt': not args.no_corrupt,
           'corruption': ccfg, 'toml': args.toml, 'checkpoint': config['checkpoint_path'],
           'check_columns': ['t', 'frame', 'bg_dE', 'person_dE', 'bg_tex', 'active'],
           'checks': [], 'onsets': [], 'offsets': [], 'corrupted_steps': [], 'episodes': [],
           'timing_columns': ['frame', 'corrupt_ms', 'sample_ms', 'decode_ms', 'detect_ms'], 'timing': []}

    def check(f, pixels):
        last = 4 * f
        if f > 0 and last // FPS == (last - 4) // FPS:
            return None
        bg, person, tex = detector(pixels[-1])
        log['checks'].append([round(last / FPS, 2), f, round(bg, 4), round(person, 4), round(tex, 4), bg > args.threshold])
        return bg > args.threshold

    def commit(latent, f, store):
        cached.buf_latent.copy_(latent)
        cached.buf_pose.copy_(pose[:, f])
        cached.buf_store.copy_(torch.as_tensor(store, device=device))
        cached.commit()

    def tick():
        torch.cuda.synchronize()
        return time.perf_counter()

    sink = stream(latents[:, 0])[0]
    detector.set_sink(sink)
    active, previous = check(0, sink[None]), False
    torch.cuda.synchronize()
    t0 = time.time()
    for frame in range(1, end):
        tc = tick()
        if active != previous:
            log['onsets' if active else 'offsets'].append([frame, round((4 * frame - 3) / FPS, 2), log['checks'][-1][0]])
        if active and not args.no_corrupt:
            if args.mode == 'commit':
                if not previous:
                    episode = Episode(corrupt, generated[:, max(1, frame - H):frame])
                    log['episodes'].append([frame, episode.info])
                f = frame - 1
                if f >= 1:
                    commit(episode(generated[:, f], pose[:, f]), f, list(cached.history_entries(f)))
            else:
                fs = list(range(frame - H, frame))
                window = torch.stack([generated[:, 0]] + [generated[:, max(f, 0)] for f in fs], dim=1)[None].clone()
                pose_window = torch.stack([pose[:, 0]] + [pose[:, max(f, 0)] for f in fs], dim=1)[None]
                ofi = torch.tensor([[0] + [f if f >= 1 else -1 for f in fs]], device=device)
                window = corrupt(window, pose_window, ofi, H)
                for i, f in enumerate(fs):
                    if f >= 1:
                        store = [SCRATCH] * H
                        store[i] = cached.history_entries(f)[i]
                        commit(window[0, :, 1 + i], f, store)
            log['corrupted_steps'].append(frame)
        previous = active

        ts = tick()
        cached.buf_latent.copy_(noise[:, frame])
        cached.buf_pose.copy_(pose[:, frame])
        row = (frame - 1) * P * m
        for band in range(P - 1, -1, -1):
            cached.set_audio([code_table], [codes], [min(frame + P - 1 - band, L - 1)],
                             audio_frames[(frame - 1) * P + P - 1 - band][None])
            for s in range(m):
                cached.buf_ints.copy_(rows[row])
                row += 1
                cached.step()
        cached.buf_store.copy_(commits[frame - 1])
        cached.commit()
        generated[:, frame] = cached.buf_latent[0]
        td = tick()
        pixels = stream(generated[:, frame])
        tx = tick()
        result = check(frame, pixels)
        te = tick()
        log['timing'].append([frame, *[round(1000 * (b - a), 3) for a, b in ((tc, ts), (ts, td), (td, tx), (tx, te))]])
        if result is not None:
            active = result
        if frame % 100 == 0:
            el = time.time() - t0
            c = log['checks'][-1]
            print(f'[{tag}] {frame}/{L - 1} {el:.0f}s {1000 * el / frame:.0f} ms/frame bg {c[2]:.2f} person {c[3]:.2f} '
                  f'tex {c[4]:.2f} corrupted {len(log["corrupted_steps"])}', flush=True)

    log['corrupted_count'] = len(log['corrupted_steps'])
    log['frames'] = end
    log['seconds'] = round(time.time() - t0, 1)
    save_file({'latents': generated[:, :end].cpu()}, str(out / f'{name}_latents.safetensors'))
    json.dump(log, open(out / f'{name}_log.json', 'w'))
    totals = np.array(log['timing'])[:, 1:].sum(0) / 1000
    print(f'[{tag}] {end} frames, onsets {log["onsets"]}, offsets {log["offsets"]}, corrupted {log["corrupted_count"]} '
          f'AR steps, {log["seconds"]}s; timing s: corrupt {totals[0]:.1f} sample {totals[1]:.1f} '
          f'decode {totals[2]:.1f} detect {totals[3]:.1f}', flush=True)
    if args.stop_frame is None:
        del detector
        torch.cuda.empty_cache()
        audio = models.mimi.decode(codes[None]).audio_values[0, 0].float().cpu().numpy()
        decode_wan22_vae(models.vae, generated[None], str(out / f'{name}.mp4'), fps=FPS, audio_np=audio, tag=tag)
    print(f'[{tag}] DONE', flush=True)


if __name__ == '__main__':
    main()

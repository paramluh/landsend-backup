"""
Universal Wide inference as the causal wavefront of docs/wan-universal.md §5, with
no caching: every round rebuilds the full [sink | H history | P present] window and
runs one forward per Euler substep.

A round: every active frontier state advances one phase band toward clean, the
one in slot 0 (t 0.2 -> 0) is emitted as a clean frame and becomes the newest
history entry, the rest move one slot left, and a fresh pure-noise frame enters at
slot P-1 while conditioned frames remain. Startup is the frontier filling up over P
rounds with zero history. With `substeps` = m, the band is crossed in m Euler steps
with every state staying inside its own band, which is exactly the t distribution
training sampled. After frame L-1 has entered, the frontier drains: no new state
joins and the remaining ones finish their bands with the right slots empty. Attention
is causal (a present slot never reads the slots to its right), so a draining round
computes exactly what a full round would, and every frame 1..L-1 is emitted.

The code table slides with the window's newest frame as in training (common.sliding_code_table), so clips
longer than AUDIO_CODE_QUANTUM // 2 frames run in full; below that the table never moves.

Audio guidance: a state's first audio_guidance_steps steps (slot P-1 is its first m) use
v = v_noaudio + audio_guidance_scale * (v_audio - v_noaudio), where v_noaudio is a second pass of the same window
with training's audio dropout at p = 1 (no_audio_tokens replace all P x (K + 1) audio tokens); scale 1.0 is off.

Samples: the pretrain-ds validation layout in `samples_dir` (see common.load_sample).
Output: <output_dir>/sample_<n>.mp4 (generation, Mimi-decoded audio muxed) and
sample_<n>_sxs.mp4 (validation sample | generation).
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.common import (
    FPS, decode_wan22_vae, find_samples, load_sample, resolve_path, run_parallel, side_by_side, sliding_code_table,
)
from inference.loader import build_models
from models.pose.audio import audio_latents_from_codes

TAG = 'pose_universal'


def forward_noaudio(net, inputs):
    """InitialLayer's training audio dropout at p = 1: the whole window's audio becomes no_audio_tokens."""
    initial = net.layers[0]
    initial.training, initial.audio_dropout_p = True, 1.0
    out = net(inputs)
    initial.training, initial.audio_dropout_p = False, 0.0
    return out


@torch.no_grad()
def generate(models, latents, pose, code_table, codes, substeps, seed, device, guidance_scale=1.0, guidance_steps=0):
    H, P = models.num_history, models.num_present
    guided = lambda pos, s: guidance_scale != 1.0 and (P - 1 - pos) * substeps + s < guidance_steps
    first = 1 + H  # first present slot: [sink | H history | P present]
    C, L, h, w = latents.shape
    latents, pose = latents.to(device), pose.to(device)
    audio = {}  # a -> audio latents of the table starting at code 2a; a only grows

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(C, L, h, w, generator=generator, device=device)  # one fixed noise per frame
    clean = [latents[:, 0]]  # frame index == list index; frame 0 is the sink
    front = []  # [frame index, state, present slot] left -> right
    ofi = torch.full((1, first + P), -1, dtype=torch.long, device=device)
    window_source = torch.zeros(1, dtype=torch.long, device=device)
    dt = -1.0 / (P * substeps)

    def window(t_scale):
        ofi.fill_(-1)
        ofi[0, 0] = 0
        x = torch.zeros(1, C, first + P, h, w, device=device)
        x[0, :, 0] = clean[0]
        history = list(range(max(1, len(clean) - H), len(clean)))
        ofi[0, first - len(history):first] = torch.tensor(history, device=device)
        for slot, frame in enumerate(history, start=first - len(history)):
            x[0, :, slot] = clean[frame]
        t = torch.zeros(1, first + P, device=device)
        for frame, state, pos in front:
            ofi[0, first + pos] = frame
            x[0, :, first + pos] = state
            t[0, first + pos] = (pos + 1) / P - t_scale
        pose_w = pose[:, ofi[0].clamp(min=0)].unsqueeze(0)  # InitialLayer masks pose where ofi < 1
        table, a = sliding_code_table(code_table, codes, front[-1][0])
        if a not in audio:
            audio.clear()
            audio[a] = audio_latents_from_codes(models.mimi, table.to(device), models.num_codebooks)
        model_ofi = torch.where(ofi >= 1, ofi - a, ofi)  # frame indices only address the audio table and masks
        return (x, t * 1000, audio[a], pose_w, model_ofi, window_source, ofi[:, first:] >= 0)

    if L < 2:
        raise RuntimeError(f'clip has {L} conditioned frames, need >= 2')
    start = time.time()
    while len(clean) < L:
        next_frame = len(clean) + len(front)
        if next_frame < L:
            front.append([next_frame, noise[:, next_frame], P - 1])
        for s in range(substeps):
            inputs = window(s / (P * substeps))
            v = models.net(inputs)[0].float()  # [C, 1 + H + P, h, w]
            if any(guided(entry[2], s) for entry in front):
                v_noaudio = forward_noaudio(models.net, inputs)[0].float()
            for entry in front:
                slot = first + entry[2]
                v_slot = v[:, slot]
                if guided(entry[2], s):
                    v_slot = v_noaudio[:, slot] + guidance_scale * (v_slot - v_noaudio[:, slot])
                entry[1] = entry[1] + v_slot * dt
        if front[0][2] == 0:
            clean.append(front.pop(0)[1])
            if len(clean) % 100 == 0:
                print(f'[{TAG}] {len(clean)}/{L} frames, {time.time() - start:.0f}s', flush=True)
        for entry in front:
            entry[2] -= 1
    elapsed = time.time() - start
    print(f'[{TAG}] {L - 1} frames in {elapsed:.1f}s: {1000 * elapsed / (L - 1):.0f} ms/frame, '
          f'RTF {elapsed / ((L - 1) * 4 / FPS):.2f}', flush=True)
    return torch.stack(clean, dim=1)  # [C, L, h, w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='pose_universal.yaml')
    parser.add_argument('--parallel', action='store_true', help='one worker per visible GPU')
    parser.add_argument('--shard', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--num-shards', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    config = yaml.safe_load(open(config_path))
    base_dir = config_path.parent
    samples_dir = Path(resolve_path(config['samples_dir'], base_dir))
    output_dir = Path(resolve_path(config['output_dir'], base_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard in (None, 0):
        shutil.copy2(config_path, output_dir / config_path.name)

    nums = find_samples(samples_dir, config['max_samples'], config['sample_seed'])
    if args.parallel:
        print(f'[{TAG}] {len(nums)} samples in {samples_dir}')
        return run_parallel(__file__, args.config, len(nums))
    tag = TAG if args.shard is None else f'{TAG}:w{args.shard}'
    if args.shard is not None:
        nums = nums[args.shard::args.num_shards]

    device = 'cuda'
    models = build_models(config, base_dir, device)
    substeps, num_steps = config['substeps'], config['num_steps']
    if num_steps != substeps * models.num_present:
        raise ValueError(f'num_steps must be substeps x num_present: {num_steps} != {substeps} x {models.num_present}; '
                         f'a frame crosses one phase band per round and P bands in total')
    guidance_scale, guidance_steps = config['audio_guidance_scale'], config['audio_guidance_steps']
    print(f'[{tag}] {len(nums)} samples, {num_steps} steps per frame ({substeps} per phase), seed {config["seed"]}, '
          f'audio guidance {guidance_scale} on the first {guidance_steps} steps')

    for n in nums:
        latents, pose, code_table, codes = load_sample(
            samples_dir, n, models.num_codebooks, config['max_latent_frames'], slide_audio=True)
        audio = models.mimi.decode(codes[None].to(device)).audio_values[0, 0].float().cpu().numpy()
        sample_path = samples_dir / f'samples_{n}.mp4'
        if not sample_path.exists():
            decode_wan22_vae(models.vae, latents[None], str(sample_path), fps=FPS, audio_np=audio, tag=tag)
        seed = config['seed'] + n  # by sample number, so sharding does not change a sample's noise
        generated = generate(models, latents, pose, code_table, codes, substeps, seed, device,
                             guidance_scale, guidance_steps)
        print(f'[{tag}] samples_{n}: {latents.shape[1]} conditioned frames -> {generated.shape[1]} latent frames')
        out_path = output_dir / f'sample_{n}.mp4'
        decode_wan22_vae(models.vae, generated[None], str(out_path), fps=FPS, audio_np=audio, tag=tag)
        side_by_side(sample_path, out_path, output_dir / f'sample_{n}_sxs.mp4', tag=tag)


if __name__ == '__main__':
    main()

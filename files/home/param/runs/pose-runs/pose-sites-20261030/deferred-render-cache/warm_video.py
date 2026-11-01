"""Keep the unchanged compiled full-wavefront video student alive between jobs.

The upstream command-line runner otherwise reloads weights and captures graphs
for every request. Only its mode-construction hook is adapted here; scheduling,
noise draws, reference selection, resolver steps, and VAE decode stay upstream.
This adapter is deliberately single-threaded: each clip owns the static buffers.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path('/home/param/vansh-video-models')


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WarmVideo:
    def __init__(self):
        sys.path[:0] = [str(REPO), str(REPO / 'production')]
        from inference import distilled
        self.runner = distilled
        self.factory = distilled.build_conditioned_modes
        self.modes = None
        self.key = None
        self.uses = 0
        self.last_mode_setup = {}
        # The process is dedicated to this one queue. The runner resolves this
        # hook at call time; all generation remains its canonical implementation.
        distilled.build_conditioned_modes = self.conditioned_modes
        self.pose_module = load_file_module('pose_sites_pose_bridge', HERE / 'bridge/transcript_pose/render.py')
        self.prepare_module = load_file_module('pose_sites_video_bridge', HERE / 'bridge/transcript_pose/render_video.py')

    @torch.no_grad()
    def conditioned_modes(self, config, latents):
        from dataloading.ds_aloe import load_actor_refs
        from utils.common import AUTOCAST_DTYPE
        key = json.dumps({
            'stabilising': config['stabilising'], 'resolving': config['resolving'],
            'audio': config['audio'], 'vae_path': config['vae_path'],
            'compile': config['compile'], 'cuda_graph': config['cuda_graph'],
            'batch_size': config['batch_size'], 'geometry': list(latents.shape[2:]),
            'type': config['type'], 'guidance': config['runtime_audio_guidance_scale'],
        }, sort_keys=True)
        if self.key is not None and key != self.key:
            raise ValueError('Warm renderer model/geometry changed; restart it to change the model contract')
        started = time.monotonic()
        reused = self.modes is not None
        if not reused:
            self.modes = self.factory(config, latents)
            self.key = key
        else:
            _, cached, resolver = self.modes
            rc = config['refs']
            refs, keep = load_actor_refs(rc['bank'], rc['actor_id'], latents[:, :1],
                                        np.random.default_rng(self.runner.reference_seed(rc['clip_key'])))
            cached.refs = (refs.transpose(0, 1)[None].cuda(), keep[None].cuda())
            # set_refs copies into the existing KV buffers used by CUDA graphs.
            with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
                resolver.set_refs(refs[None].cuda(), keep[None].cuda())
            # start_clip refreshes reference/sink/history caches and clears audio
            # conditioning. Zero the wave scratch to make the boundary explicit.
            cached.wave_latent.zero_()
            cached.wave_pose.zero_()
            cached.wave_audio.zero_()
            cached.wave_valid.zero_()
        self.uses += 1
        torch.cuda.synchronize()
        self.last_mode_setup = {
            'weights_and_graphs_reused': reused, 'engine_use': self.uses,
            'mode_setup_seconds': round(time.monotonic() - started, 4),
            'schedule': 'full-wavefront', 'present_frames_per_pass': 4,
            'stabilizer_steps_per_band': 2, 'resolver_steps': 1,
        }
        print('WARM_VIDEO_MODES', json.dumps(self.last_mode_setup), flush=True)
        return self.modes

    def render_pose(self, sample, frame0_path):
        from types import SimpleNamespace
        from safetensors.torch import load_file
        sample, frame0_path = Path(sample), Path(frame0_path)
        frame0 = load_file(str(frame0_path))['frame0_pose'].float().numpy()
        if frame0.shape != (133, 3) or not np.isfinite(frame0).all():
            raise ValueError('Expected finite frame0 [133,3] y,x,confidence')
        codes = load_file(str(sample / 'pose_tokens.safetensors'))['codes']
        args = SimpleNamespace(out=sample, pose_latents=True, skeleton_video=False, video_sample=False)
        result = self.pose_module.render_outputs(args, codes, frame0)
        (sample / 'render.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    def render_video(self, args):
        config_path, receipt = self.prepare_module.prepare(args)
        import yaml
        config = yaml.safe_load(config_path.read_text())
        self.runner.run(config)
        status_path = Path(config['output_dir']) / 'status.json'
        status = json.loads(status_path.read_text())
        status.update(self.last_mode_setup)
        self.runner.write_json(status_path, status)
        return status

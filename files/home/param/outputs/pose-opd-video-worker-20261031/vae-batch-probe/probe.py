import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

SPOOL = Path('/home/param/outputs/pose-opd-video-worker-20261031')
sys.path[:0] = [str(SPOOL / 'vae-batch-probe'),
    '/home/param/pose-worktrees/pose-opd-live-demo/infer/pose_opd_live/renderer/amu/vae']
import fast_decode as native
import video_decode as adapter
from models.autoencoders.wanvae import Wan2_2_VAE

torch.set_num_threads(2)
torch.set_grad_enabled(False)
started = time.monotonic()
def report(phase, **values):
    print(json.dumps({'phase': phase, 'elapsed': time.monotonic() - started, **values}), flush=True)

vae = Wan2_2_VAE(vae_pth='/home/param/models/wan22-vae/Wan2.2_VAE.pth', dtype=torch.float32, device='cuda')
gt = load_file('/home/param/runs/pose-runs/pose-opd-credit-step89-live/people/clip-02/gt/video_latents.safetensors')['latents'].float()
z = torch.stack([gt[(torch.arange(64) + offset) % len(gt)] for offset in (0, 7, 14, 21)]).cuda()
report('weights_loaded', source_latents=len(gt))
serial = native.StreamDecoder(vae, mode='fp32', compile=True, graph=True, compile_mode='default')
report('serial_ready')
batch = adapter.BatchedStreamDecoder(vae, batch_size=4)
report('batch_ready')

# Compare pre-quantization convolution outputs using the same native eager
# arithmetic/cache state. Keep the graphs unchanged for the real timing test.
saved_native, saved_adapter = native.postprocess, adapter.postprocess
old_serial, old_batch = (serial.graph, serial.fn), (batch.graph, batch.fn)
native.postprocess = lambda y: y.float()
adapter.postprocess = lambda y: y.float()
serial.graph, serial.fn = None, serial.st.step
batch.graph, batch.fn = None, batch.st.step
try:
    serial_raw = []
    for row in range(4):
        serial.reset()
        serial_raw.append([serial.push_gpu(z[row, t]).cpu() for t in range(4)])
    batch.reset()
    raw_max, raw_sse, raw_count = 0., 0., 0
    for t in range(4):
        y = batch.push_gpu(z[:, t]).cpu()
        ref = torch.cat([serial_raw[row][t] for row in range(4)])
        delta = (y-ref).double()
        raw_max = max(raw_max, delta.abs().max().item())
        raw_sse += delta.square().sum().item()
        raw_count += delta.numel()
    report('float_parity', max_absolute=raw_max, rmse=math.sqrt(raw_sse/raw_count))
finally:
    native.postprocess, adapter.postprocess = saved_native, saved_adapter
    serial.graph, serial.fn = old_serial
    batch.graph, batch.fn = old_batch

torch.cuda.synchronize()
begin = time.monotonic()
reference = []
for row in range(4):
    serial.reset()
    reference.append([serial.push(z[row, t]) for t in range(64)])
serial_seconds = time.monotonic() - begin
report('serial_done', seconds=serial_seconds)
batch.reset()
begin = time.monotonic()
actual = [batch.push(z[:, t]) for t in range(64)]
batch_seconds = time.monotonic() - begin
report('batch_done', seconds=batch_seconds)

pixel_max, pixel_sse, pixel_count, different = 0, 0., 0, 0
for t in range(64):
    ref = np.stack([reference[row][t] for row in range(4)])
    delta = actual[t].astype(np.int16) - ref.astype(np.int16)
    pixel_max = max(pixel_max, int(np.abs(delta).max()))
    pixel_sse += np.square(delta.astype(np.float64)).sum()
    pixel_count += delta.size
    different += np.count_nonzero(delta)
psnr = float('inf') if pixel_sse == 0 else 10 * math.log10(255**2/(pixel_sse/pixel_count))

# A permuted second clip must not inherit another row's previous cache state.
order = [3, 1, 0, 2]
batch.reset()
permutation_max = 0
for t in range(4):
    repeated = batch.push(z[order, t])
    permutation_max = max(permutation_max, int(np.abs(repeated.astype(np.int16)-actual[t][order].astype(np.int16)).max()))
result = dict(serial_seconds=serial_seconds, batch_seconds=batch_seconds,
    speedup=serial_seconds/batch_seconds, latent_frames_per_row=64,
    raw_max_absolute=raw_max, raw_rmse=math.sqrt(raw_sse/raw_count),
    max_pixel_difference=pixel_max, pixel_psnr_db=psnr,
    pixel_difference_fraction=different/pixel_count, permutation_max_pixel_difference=permutation_max,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
report('complete', **result)
(SPOOL/'vae-batch-probe/result.json').write_text(json.dumps(result, indent=2)+'\n')

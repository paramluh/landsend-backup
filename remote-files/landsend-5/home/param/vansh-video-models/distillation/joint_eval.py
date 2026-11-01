"""Bounded paired-noise sensitivity probes for joint nine-reference DMD.

The six startup frames and every measured frame are generated. Only
the supplied sink and reference bank are visual ground truth. Evaluation uses
the training generator's continuous wavefront, including its cache; it does not
restart a short rollout at each four-frame boundary.

History interventions select the final three startup frames (generated frames
4, 5, and 6), entering when the selected frame is first committed to history.
Sink/reference interventions enter before startup, so
their measurements include propagation through the generated startup prefix.
The two intervention types intentionally answer different causal questions.

Call only on an unsharded student, ordinarily on rank zero after saving a
checkpoint. The caller writes the returned JSON-safe object beside that
checkpoint and supplies a fixed evaluation row, without consuming training
sampler state. This module has no optimizer or distributed collectives.
"""

from contextlib import contextmanager
import math
import random

import numpy as np
import torch

from distillation.joint_dmd import generate_stabilizer


def _indices(config, name, default, maximum):
    values = list(config.get(name, default))
    if len(values) != len(set(values)) or any(type(v) is not int or not 0 <= v < maximum for v in values):
        raise ValueError(f'{name} must contain distinct indices in [0,{maximum})')
    return values


def _rms(value, dims):
    return value.float().square().mean(dim=dims).sqrt()


def _finite(value):
    """Keep failures visible while producing strict JSON (never NaN/Infinity)."""
    if isinstance(value, torch.Tensor):
        return _finite(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextmanager
def _evaluation_state(net, device):
    """Restore every module mode and the RNGs even if a probe raises."""
    modes = [(module, module.training) for module in net.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            net.eval()
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _seed_forward(seed, device):
    # Seed just the generators covered by _evaluation_state. torch.manual_seed
    # would also reseed other GPUs on a multi-GPU host.
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.default_generators[index].manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))


def _direction(shape, device, seed, rms):
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    scale = _rms(noise, tuple(range(1, noise.ndim))).clamp_min(1e-12)
    return noise * (rms / scale).reshape(-1, *([1] * (noise.ndim - 1)))


def _weighted_mean(values, weights):
    """Input has already excluded padding; do not hide nonfinite valid rows."""
    return (values * weights.reshape(-1, *([1] * (values.ndim - 1)))).sum(0) / weights.sum()


def _measure(baseline, perturbed, input_rms, weights, active, ofi, prefix_frames):
    active = active & (weights > 0) & (input_rms > 0)
    indices = active.nonzero(as_tuple=False).flatten()
    if not len(indices):
        return {'status': 'skipped_no_active_rows', 'active_rows': [], 'frames': []}
    base, changed = baseline[indices], perturbed[indices]
    w, injected = weights[indices], input_rms[indices]
    delta = _rms(changed - base, (1, 3, 4))
    base_rms = _rms(base, (1, 3, 4))
    gain = delta / injected[:, None]
    relative = delta / base_rms.clamp_min(1e-12)
    mean_delta, mean_gain, mean_relative, mean_base = [
        _weighted_mean(value, w) for value in (delta, gain, relative, base_rms)
    ]
    finite = torch.isfinite(base).flatten(3).all(3).all(1) & torch.isfinite(changed).flatten(3).all(3).all(1)
    frames = [
        {'target_index': frame + 1, 'generated_frame_index': frame + prefix_frames + 1,
         'original_frame_indices': ofi[indices, frame],
         'output_delta_rms': mean_delta[frame], 'amplification': mean_gain[frame],
         'relative_output_change': mean_relative[frame], 'baseline_rms': mean_base[frame],
         'nonfinite_rows': int((~finite[:, frame]).sum().item())}
        for frame in range(base.shape[2])
    ]
    return {
        'status': 'ok' if bool(finite.all()) else 'nonfinite_output',
        'active_rows': indices, 'sample_weights': w, 'input_rms_per_row': injected,
        'frames': frames, 'output_delta_rms_per_row': delta, 'amplification_per_row': gain,
        'maximum_amplification': mean_gain.max(), 'last_frame_amplification': mean_gain[-1],
    }


def run_sensitivity(net, rows, config):
    """Return per-frame latent RMS response to fixed, small input perturbations.

    ``rows`` uses the prepared training representation, but has ``7 + N``
    temporal entries for ``N = sensitivity_frames``: sink, six context poses,
    and N targets. generate_stabilizer uses all six context poses to generate
    its six startup frames, then generates N targets. Temporal visual GT is
    never read. Only the last three startup frames (4..6) are history probe
    sites; their indices 0..2 match the critic's three history slots.

    Config keys: sensitivity_frames (8, a positive multiple of four),
    sensitivity_seeds ([1729]), sensitivity_rms (.01 latent units),
    sensitivity_history_indices ([0,1,2], oldest first),
    sensitivity_reference_indices ([0,4,8]), band_steps ([2,2,2,2]), and
    stabilizer_prefix_frames (6, fixed to the student training prefix).
    A longer horizon or all nine reference indices is an explicit cost choice.
    """
    if getattr(net, 'shards', None) is not None:
        raise ValueError('Sensitivity evaluation requires an unsharded student; do not call a sharded net on rank zero')
    prefix_frames = config.get('stabilizer_prefix_frames', 6)
    if type(prefix_frames) is not int or prefix_frames != 6:
        raise ValueError('Sensitivity evaluation must use the student six-frame generated prefix')
    frames = config.get('sensitivity_frames', 8)
    if type(frames) is not int or frames < 4 or frames % 4:
        raise ValueError('sensitivity_frames must be a positive multiple of four')
    magnitude = float(config.get('sensitivity_rms', .01))
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError('sensitivity_rms must be finite and positive')
    seeds = list(config.get('sensitivity_seeds', [1729]))
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError('sensitivity_seeds must contain nonnegative signed-64-bit integers')
    history_indices = _indices(config, 'sensitivity_history_indices', [0, 1, 2], 3)
    reference_indices = _indices(config, 'sensitivity_reference_indices', [0, 4, 8], 9)
    band_steps = config.get('band_steps', [2, 2, 2, 2])
    latent, refs = rows['latents'], rows['refs']
    if latent.ndim != 5 or latent.shape[2] != 7 + frames:
        raise ValueError(f'Sensitivity rows must contain sink + six context positions + {frames} target positions')
    batch, channels, _, height, width = latent.shape
    if tuple(refs.shape) != (batch, channels, 9, height, width):
        raise ValueError('Sensitivity rows require nine reference latents of the training spatial shape')
    device = latent.device
    weights = rows.get('sample_weight', rows.get('weight', torch.ones(batch, device=device))).to(
        device=device, dtype=torch.float32)
    if tuple(weights.shape) != (batch,) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('Sensitivity sample weights must be finite nonnegative [B] values')
    if not bool((weights > 0).any()):
        return {'format': 'joint-latent-sensitivity-v1', 'status': 'skipped_no_valid_rows'}
    keep = rows.get('ref_keep', torch.ones(batch, 9, dtype=torch.bool, device=device)).to(device=device, dtype=torch.bool)
    if tuple(keep.shape) != (batch, 9):
        raise ValueError('Sensitivity reference validity mask must have shape [B,9]')

    # The caller prepares evaluation rows without stochastic training dropout.
    # Both generator calls explicitly disable prefix-history dropout so only
    # the requested perturbation differs between the paired rollouts.
    # Preserve its semantic bad-audio mask rather than enabling invalid audio.
    fixed_rows = dict(rows)
    fixed_rows['drop'] = rows.get('drop', torch.zeros(batch, dtype=torch.bool, device=device)).to(
        device=device, dtype=torch.bool)
    if tuple(fixed_rows['drop'].shape) != (batch,):
        raise ValueError('Sensitivity audio mask must have shape [B]')
    cases = [('history', index) for index in history_indices] + [('sink', None)]
    cases += [('reference', index) for index in reference_indices]
    result = {
        'format': 'joint-latent-sensitivity-v1', 'status': 'ok',
        'generated_prefix_frames': prefix_frames, 'measured_target_frames': frames,
        'total_generated_frames_per_rollout': frames + prefix_frames,
        'endpoint_t': float(rows.get('history_t', .2)), 'band_steps': list(band_steps),
        'seeds': seeds, 'requested_input_rms': magnitude,
        'input_rms_convention': 'RMS over C,H,W of the one perturbed conditioning frame; no dilution by unperturbed frames',
        'amplification_definition': 'RMS(perturbed_output - baseline_output) / input_perturbation_RMS, then weighted mean over valid rows',
        'perturbation_direction': 'The same RMS-normalized Gaussian direction per row and seed at every conditioning site',
        'history_intervention': 'One of the last three generated startup frames (4..6) is perturbed at history commitment; its earlier present-slot states stay unchanged',
        'sink_reference_intervention': 'Perturbed from startup, including effects mediated through generated startup history',
        'output_space': 'Stabilizer latent endpoints at history_t; not decoded pixels or resolver output',
        'reference_indices': reference_indices, 'history_indices_oldest_first': history_indices,
        'history_probe_generated_frame_indices': [prefix_frames - 2 + index for index in history_indices],
        'sample_weights': weights, 'audio_dropped_rows': fixed_rows['drop'],
        'target_original_frame_indices': rows['ofi'][:, -frames:],
        'rollouts': [],
    }
    with _evaluation_state(net, device):
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((batch, channels, frames + prefix_frames, height, width), device=device,
                                dtype=torch.float32, generator=generator)
            _seed_forward(seed, device)
            baseline, baseline_context = generate_stabilizer(
                net, fixed_rows, band_steps, noise=noise.clone(), num_targets=frames,
                prefix_frames=prefix_frames, history_dropout=())
            baseline = baseline.detach()
            direction = _direction((batch, channels, height, width), device,
                                   (seed + 1_000_003) % (2**63), magnitude)
            trial = {'seed': seed, 'baseline_rms_per_row': _rms(baseline, (1, 3, 4)),
                     'generated_history_rms_per_row': _rms(baseline_context['history'], (1, 3, 4)),
                     'cases': {}}
            for kind, index in cases:
                delta = direction.clone()
                changed_rows = dict(fixed_rows)
                history_delta = None
                active = weights > 0
                if kind == 'history':
                    history_delta = torch.zeros((batch, channels, 3, height, width), device=device, dtype=torch.float32)
                    history_delta[:, :, index] = delta
                elif kind == 'sink':
                    changed_rows['latents'] = latent.float().clone()
                    changed_rows['latents'][:, :, 0] += delta
                    delta = changed_rows['latents'][:, :, 0] - latent[:, :, 0].float()
                else:
                    active = active & keep[:, index]
                    delta = delta * keep[:, index, None, None, None]
                    changed_rows['refs'] = refs.float().clone()
                    changed_rows['refs'][:, :, index] += delta
                    delta = changed_rows['refs'][:, :, index] - refs[:, :, index].float()
                key = kind if index is None else f'{kind}_{index}'
                if not bool(active.any()):
                    trial['cases'][key] = {'status': 'skipped_no_active_rows', 'active_rows': [], 'frames': []}
                    continue
                _seed_forward(seed, device)
                changed, _ = generate_stabilizer(
                    net, changed_rows, band_steps, noise=noise.clone(), num_targets=frames,
                    prefix_frames=prefix_frames, history_perturbation=history_delta,
                    history_dropout=())
                measured = _measure(baseline, changed, _rms(delta, (1, 2, 3)), weights, active,
                                    rows['ofi'][:, -frames:], prefix_frames)
                trial['cases'][key] = measured
                if measured['status'] == 'nonfinite_output':
                    result['status'] = 'nonfinite_output'
                del changed
            result['rollouts'].append(trial)
    return _finite(result)

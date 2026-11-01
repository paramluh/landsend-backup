"""Paired endpoint context sensitivity with frozen generated context replay.

The baseline is a continuous causal wavefront with six generated startup frames.
Four panels start from the first steady query containing four evaluation targets,
respectively at t=.4,.6,.8,1. Only the selected target is reintegrated after an
intervention. Other present/history latent trajectories are replayed verbatim;
all of their network representations are recomputed. This intentionally differs
from the legacy sensitivity evaluator's autoregressive propagation experiment.
"""

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

import torch

from distillation.full_gt_dmd import _plan
from distillation.joint_eval import _direction, _evaluation_state, _finite, _rms, _seed_forward

PROTOCOL_VERSION = 'generated-context-endpoint-replay-v1'
CATEGORIES = {
    'stabilizer': ['previous 1', 'previous 2', 'previous 3', 'sink', 'reference bank mean'],
    'resolver': ['previous 1', 'previous 2', 'sink', 'reference bank mean'],
}
_RESOLVER_PLAN = [(0, [[('f', 3), ('r', 1, 1), ('r', 2, 2), ('f', 0)]]),
                  (1, [[('f', 3), ('f', 1)], [('f', 3), ('f', 2)]]),
                  (3, [[('f', 3)]])]


@contextmanager
def _eager_blocks(net):
    """Do not add evaluation-only plans/batch sizes to the training block cache."""
    override = hasattr(net, 'block_fn') and hasattr(net, '_block')
    previous = net.block_fn if override else None
    try:
        if override:
            net.block_fn = net._block
        yield
    finally:
        if override:
            net.block_fn = previous


def _hash(value):
    value = value.detach().cpu()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _query(net, query, frames, tokens):
    _seed_forward(query['rng_seed'], frames[0][0].device)
    audio = net.frame_audio(tokens, query['ofi'], query['drop'])
    velocity, _, _ = net.forward_pass(frames, query['plan'], None, audio,
                                      query['audio_mask'], query['heads'])
    return velocity


def _bind(net, rows, seed):
    _seed_forward(seed, rows['latents'].device)
    return net.references(rows['refs'], rows['ref_keep'])


def _wavefront(net, rows, seed, steps):
    """Explicit temporal recomputation avoids stale sink/history K/V on replay."""
    latent, pose, ofi = rows['latents'], rows['pose'], rows['ofi']
    batch, channels, _, height, width = latent.shape
    total, prefix, dt = 10, 6, .2 / steps
    generator = torch.Generator(device=latent.device).manual_seed(seed)
    noise = torch.randn((batch, channels, total, height, width), device=latent.device,
                        dtype=torch.float32, generator=generator)
    active, completed, queries = {}, {}, {}
    zero = torch.zeros_like(latent[:, :, 0], dtype=torch.float32)
    all_valid = torch.ones(batch, dtype=torch.bool, device=latent.device)
    with _bind(net, rows, seed):
        tokens = net.audio_tokens(rows['audio'])
        serial = 0
        for first in range(-2, total + 1):
            entering = first + 3
            if 1 <= entering <= total:
                active[entering] = noise[:, :, entering - 1]
            leaving = first - 1
            if leaving in active:
                completed[leaving] = active.pop(leaving)
            for substep in range(steps):
                frames, frame_ids, indices, masks = [], [], [], []
                for slot in range(4):
                    identity = first + slot
                    valid = all_valid if identity in active else ~all_valid
                    time = (.2 + .2 * (slot + 1) - dt * substep) * valid.float()
                    frames.append((active.get(identity, zero), pose[:, :, identity] if identity in active else None,
                                   time, 4 + slot if identity in active else 3, valid))
                    frame_ids.append(identity if identity in active else None)
                    indices.append(ofi[:, min(max(identity, 0), total)])
                    masks.append(valid)
                for identity in range(first - 3, first):
                    valid = all_valid if identity in completed else ~all_valid
                    frames.append((completed.get(identity, zero), pose[:, :, identity] if identity in completed else None,
                                   .2 * valid.float(), 3, valid))
                    frame_ids.append(identity if identity in completed else None)
                frames.append((latent[:, :, 0].float(), None, 0., 0, all_valid))
                frame_ids.append(0)
                # Startup padding is a self/sink/reference context token at
                # history slot 3, exactly like the production cached pad. A
                # padded present slot must not accumulate three additional
                # predecessor inputs through deeper layers.
                plan = []
                for start, specs in _plan(4):
                    for offset, spec in enumerate(specs):
                        position = start + offset
                        if position < 4 and frame_ids[position] is None:
                            spec = [('f', 7), ('f', position)]
                        else:
                            spec = [('r', entry[1], 4 + entry[1])
                                    if entry[0] == 'f' and entry[1] < 4 and frame_ids[entry[1]] is None
                                    else entry for entry in spec]
                        plan.append((position, [spec]))
                query = {'frames': frames, 'frame_ids': frame_ids, 'plan': plan, 'heads': 4,
                         'ofi': torch.stack(indices, 1), 'audio_mask': torch.stack(masks, 1),
                         'drop': rows['drop'], 'rng_seed': (seed + 10_000 + serial) % (2**63), 'dt': dt}
                serial += 1
                velocity = _query(net, query, frames, tokens)
                if first >= prefix + 1:
                    queries[first, substep] = query
                for slot in range(4):
                    identity = first + slot
                    if identity in active:
                        active[identity] = active[identity] - dt * velocity[:, :, slot]
        completed[total] = active[total]
    return queries, completed


def _resolver_baseline(net, rows, stabilized, seed):
    """Actual student one-step resolver, fed generated stabilized endpoints."""
    resolved, measured = {}, None
    batch = rows['latents'].shape[0]
    valid = torch.ones(batch, dtype=torch.bool, device=rows['latents'].device)
    zero = torch.zeros_like(rows['latents'][:, :, 0], dtype=torch.float32)
    with _bind(net, rows, seed):
        tokens = net.audio_tokens(rows['audio'])
        for identity in range(1, 8):
            frames = [(stabilized[identity], rows['pose'][:, :, identity], .2, 3, valid)]
            ids = [identity]
            for previous in (identity - 2, identity - 1):
                exists = previous in resolved
                frames.append((resolved.get(previous, zero), rows['pose'][:, :, previous] if exists else None,
                               0., 2, valid if exists else ~valid))
                ids.append(previous if exists else None)
            frames.append((rows['latents'][:, :, 0].float(), None, 0., 0, valid))
            ids.append(0)
            query = {'frames': frames, 'frame_ids': ids, 'plan': _RESOLVER_PLAN, 'heads': 1,
                     'ofi': rows['ofi'][:, identity:identity+1], 'audio_mask': valid[:, None],
                     'drop': rows['drop'], 'rng_seed': (seed + 20_000 + identity) % (2**63), 'dt': .2}
            velocity = _query(net, query, frames, tokens)
            resolved[identity] = stabilized[identity] - .2 * velocity[:, :, 0]
            if identity == 7:
                measured = query
    return measured, resolved[7]


def _replay(net, rows, trace, target, seed, intervention=None, direction=None, sign=1):
    """Replay all non-target latent values; fresh reference and temporal caches."""
    changed_rows = dict(rows)
    fixed_delta = None
    if intervention is not None and intervention[0] == 'reference':
        index = intervention[1]
        changed_rows['refs'] = rows['refs'].float().clone()
        active = rows['ref_keep'][:, index, None, None, None]
        changed_rows['refs'][:, :, index] += sign * direction * active
        fixed_delta = changed_rows['refs'][:, :, index] - rows['refs'][:, :, index].float()
    first = trace[0]
    slot = first['frame_ids'].index(target)
    value = first['frames'][slot][0].clone()
    injected = []
    with _bind(net, changed_rows, seed):
        tokens = net.audio_tokens(rows['audio'])
        for query in trace:
            frames = list(query['frames'])
            slot = query['frame_ids'].index(target)
            frames[slot] = (value, *frames[slot][1:])
            if intervention is not None:
                if intervention[0] == 'reference':
                    delta = fixed_delta
                else:
                    identity = 0 if intervention[0] == 'sink' else target - intervention[1]
                    index = query['frame_ids'].index(identity)
                    original = frames[index][0].float()
                    valid = frames[index][4]
                    delta_requested = sign * direction * valid[:, None, None, None]
                    changed = original + delta_requested
                    delta = changed - original
                    frames[index] = (changed, *frames[index][1:])
                injected.append(_rms(delta, (1, 2, 3)))
            velocity = _query(net, query, frames, tokens)
            value = value - query['dt'] * velocity[:, :, slot]
    # Floating-point additions can produce tiny differences at different
    # baseline values. Measure the actual injection over all replayed queries.
    actual = torch.stack(injected, 1) if injected else None
    denominator = actual.square().mean(1).sqrt() if actual is not None else None
    return value, denominator, actual


def _summary(records, category, weights):
    samples = []
    for record in records:
        for sample in record['samples']:
            if sample['category'] == category and sample['gain'] is not None:
                samples.append((sample['gain'], float(weights[sample['row']])))
    if not samples:
        return {'mean': None, 'std': None, 'count': 0, 'weight': 0.}
    mass = sum(weight for _, weight in samples)
    mean = sum(value * weight for value, weight in samples) / mass
    variance = sum(weight * (value - mean)**2 for value, weight in samples) / mass
    return {'mean': mean, 'std': math.sqrt(max(variance, 0.)), 'count': len(samples), 'weight': mass}


def _measure_panel(net, rows, trace, target, seed, direction, objective, expected):
    baseline, _, _ = _replay(net, rows, trace, target, seed)
    replay_error = _rms(baseline - expected, (1, 2, 3))
    if not bool(torch.isfinite(baseline).all()):
        raise FloatingPointError('Nonfinite baseline sensitivity endpoint')
    weights = rows['sample_weight']
    predecessors = 3 if objective == 'stabilizer' else 2
    cases = [('history', distance) for distance in range(1, predecessors + 1)] + [('sink', None)]
    cases += [('reference', index) for index in range(rows['refs'].shape[2])]
    measured, samples = [], []
    ref_gains = [[] for _ in range(len(weights))]
    for intervention in cases:
        kind, index = intervention
        active = weights > 0
        if kind == 'reference':
            active = active & rows['ref_keep'][:, index]
        trials, gains = [], []
        if not bool(active.any()):
            continue
        for sign in (1, -1):
            changed, injected, actual = _replay(net, rows, trace, target, seed, intervention, direction, sign)
            response = _rms(changed - baseline, (1, 2, 3))
            valid = active & (injected > 0)
            if not bool(torch.isfinite(changed[active]).all()) or not bool(torch.isfinite(injected[active]).all()):
                raise FloatingPointError('Nonfinite endpoint sensitivity trial')
            gain = response / injected.clamp_min(1e-30)
            gain = torch.where(valid, gain, torch.full_like(gain, float('nan')))
            gains.append(gain)
            trials.append({'sign': sign, 'input_rms_per_row': injected, 'input_rms_per_query': actual,
                           'endpoint_delta_rms_per_row': response, 'gain_per_row': gain})
        paired = torch.stack(gains).mean(0)
        measured.append({'kind': kind, 'index': index, 'trials': trials, 'paired_gain_per_row': paired})
        category = f'previous {index}' if kind == 'history' else 'sink'
        for row in active.nonzero(as_tuple=False).flatten().tolist():
            value = float(paired[row])
            if not math.isfinite(value):
                continue
            if kind == 'reference':
                ref_gains[row].append(value)
            else:
                samples.append({'row': row, 'category': category, 'gain': value})
    for row, values in enumerate(ref_gains):
        if values:
            samples.append({'row': row, 'category': 'reference bank mean',
                            'gain': sum(values) / len(values), 'valid_references': len(values)})
    return {'seed': seed, 'baseline_rms_per_row': _rms(baseline, (1, 2, 3)),
            'baseline_replay_error_rms_per_row': replay_error, 'cases': measured, 'samples': samples}


def run_endpoint_sensitivity(net, rows, config):
    """Return strict-JSON measurements for four stabilizer slots and one resolver.

    ``rows`` has prepared sink + six startup poses + at least four target poses,
    nine references and audio latents. Only the sink/reference image values are
    read. Temporal GT, including targets, never conditions this evaluation.
    ``sensitivity_seeds``, ``sensitivity_rms`` and equal ``band_steps`` are fixed
    protocol settings; optional ``sensitivity_validate_half_rms`` repeats the
    protocol at half amplitude for its initial numerical validation.
    """
    if getattr(net, 'shards', None) is not None:
        raise ValueError('Endpoint sensitivity requires an unsharded student')
    latent, refs = rows['latents'], rows['refs']
    if latent.ndim != 5 or latent.shape[2] < 11 or rows['pose'].shape != latent.shape:
        raise ValueError('Endpoint sensitivity requires sink + six startup + at least four target positions')
    batch, channels, _, height, width = latent.shape
    if refs.shape != (batch, channels, 9, height, width):
        raise ValueError('Endpoint sensitivity requires exactly nine supplied references')
    if rows['ofi'].shape != (batch, latent.shape[2]):
        raise ValueError('Endpoint sensitivity frame identities must match temporal inputs')
    keep = rows['ref_keep']
    if keep.shape != (batch, 9) or keep.dtype != torch.bool:
        raise ValueError('Reference validity must be bool [B,9]')
    weights = rows.get('sample_weight', rows.get('weight', torch.ones(batch, device=latent.device))).to(
        device=latent.device, dtype=torch.float32)
    if weights.shape != (batch,) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('Sample weights must be finite nonnegative [B] values')
    if not bool((weights > 0).any()):
        return {'format': PROTOCOL_VERSION, 'status': 'skipped_no_valid_rows', 'panels': []}
    if not bool((rows['ofi'][weights > 0, 1:11] >= 1).all()):
        raise ValueError('Sensitivity panel needs valid contiguous generated startup and target positions')
    if not bool(((rows['ofi'][weights > 0, 2:11] - rows['ofi'][weights > 0, 1:10]) == 1).all()):
        raise ValueError('Sensitivity positions must be consecutive source frames')
    seeds = list(config.get('sensitivity_seeds', [1729]))
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError('Sensitivity seeds must be distinct nonnegative signed-64-bit integers')
    magnitude = float(config.get('sensitivity_rms', .01))
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError('Sensitivity RMS must be finite and positive')
    bands = list(config.get('band_steps', [2, 2, 2, 2]))
    if len(bands) != 4 or len(set(bands)) != 1 or any(type(value) is not int or value < 1 for value in bands):
        raise ValueError('Sensitivity requires four equally subdivided stabilizer bands')
    fixed = dict(rows)
    fixed['sample_weight'] = weights
    fixed['drop'] = rows.get('drop', torch.zeros(batch, dtype=torch.bool, device=latent.device))
    if fixed['drop'].shape != (batch,) or fixed['drop'].dtype != torch.bool:
        raise ValueError('Audio dropout must be bool [B]')
    result = {'format': PROTOCOL_VERSION, 'status': 'ok', 'requested_input_rms': magnitude,
              'seeds': seeds, 'band_steps': bands, 'generated_startup_frames': 6,
              'measured_stabilizer_targets': 4, 'measured_resolver_targets': 1,
              'context_protocol': 'generated baseline trajectories replayed; only selected target reintegrated',
              'whole_block_compile': False,
              'reference_protocol': 'paired per-valid-reference gains averaged within row before across rows/seeds',
              'input_rms_convention': 'actual injected C,H,W RMS, quadratic mean over remaining target queries',
              'sample_weights': weights, 'target_original_frame_indices': rows['ofi'][:, 7:11],
              'conditioning_hashes': [{'sink': _hash(latent[row, :, 0]),
                                       'pose': _hash(rows['pose'][row, :, :11]),
                                       'audio': _hash(rows['audio'][row])} for row in range(batch)],
              'reference_hashes': [[_hash(refs[row, :, index]) for index in range(9)] for row in range(batch)],
              'reference_valid': keep, 'panels': []}
    panels = [{'objective': 'stabilizer', 'slot': slot, 'start_t': .4 + .2 * slot, 'endpoint_t': .2,
               'target_position': 7 + slot, 'remaining_steps': (slot + 1) * bands[0], 'records': []}
              for slot in range(4)]
    panels.append({'objective': 'resolver', 'slot': 0, 'start_t': .2, 'endpoint_t': 0.,
                   'target_position': 7, 'remaining_steps': 1, 'records': []})
    with _evaluation_state(net, latent.device), _eager_blocks(net):
        for seed in seeds:
            queries, completed = _wavefront(net, fixed, seed, bands[0])
            resolver, resolved = _resolver_baseline(net, fixed, completed, seed)
            direction = _direction((batch, channels, height, width), latent.device,
                                   (seed + 1_000_003) % (2**63), magnitude)
            for panel in panels:
                target = panel['target_position']
                trace = ([queries[first, substep] for first in range(7, target + 1)
                          for substep in range(bands[0])] if panel['objective'] == 'stabilizer' else [resolver])
                expected = completed[target] if panel['objective'] == 'stabilizer' else resolved
                record = _measure_panel(net, fixed, trace, target, seed, direction, panel['objective'], expected)
                panel['records'].append(record)
        for panel in panels:
            panel['categories'] = CATEGORIES[panel['objective']]
            panel['summary'] = {category: _summary(panel['records'], category, weights)
                                for category in panel['categories']}
        result['panels'] = panels
    result = _finite(result)
    if config.get('sensitivity_validate_half_rms', False):
        half_config = dict(config, sensitivity_rms=magnitude / 2, sensitivity_validate_half_rms=False)
        result['half_amplitude_validation'] = run_endpoint_sensitivity(net, rows, half_config)
    return result


def render_endpoint_sensitivity(result, output_dir):
    """Render all slots on one gain scale, including the separate resolver plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # Share the numeric limits across both figures, with room for error bars
    # and count labels. Independent autoscaling obscures differences by slot.
    upper = max((summary['mean'] + (summary['std'] or 0.)
                 for panel in result.get('panels', [])
                 for summary in panel['summary'].values()
                 if summary['mean'] is not None), default=0.)
    gain_limits = (0., upper * 1.18 if upper > 0. else 1.)
    for objective in ('stabilizer', 'resolver'):
        panels = [panel for panel in result.get('panels', []) if panel['objective'] == objective]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.5),
                                 squeeze=False, sharey=True)
        for ax, panel in zip(axes[0], panels):
            categories = panel['categories']
            for index, category in enumerate(categories):
                summary = panel['summary'][category]
                if summary['mean'] is not None:
                    ax.bar(index, summary['mean'], color='#4579b4' if index < len(categories)-1 else '#e59048',
                           yerr=summary['std'], capsize=3)
                    ax.annotate(f"n={summary['count']}", (index, summary['mean']), xytext=(0, 7),
                                textcoords='offset points', ha='center', fontsize=8)
            ax.axvline(len(categories)-1.5, color='0.65', linestyle='--', linewidth=1)
            ax.set_xticks(range(len(categories)), categories, rotation=35, ha='right')
            if ax is axes[0, 0]:
                ax.set_ylabel('Endpoint perturbation gain (latent RMS / latent RMS)')
            ax.set_ylim(gain_limits)
            ax.set_title(f"{objective.title()} slot {panel['slot'] + 1} · t={panel['start_t']:.2f} → {panel['endpoint_t']:.2f}")
            ax.grid(axis='y', alpha=.2)
        fig.suptitle('Endpoint sensitivity with replayed generated context')
        fig.text(.5, .01, f"Paired ± perturbations, requested RMS {result['requested_input_rms']:g}; "
                 'bars: mean, whiskers: standard deviation across clips/seeds; references averaged within each clip.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .94))
        name = f'{objective}_context_gain'
        path = output_dir / f'{name}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = path
    return paths

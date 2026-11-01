"""Detached diagnostics and the explicit online training scalar contract."""
import math

import torch
import torch.nn.functional as F


OBJECTIVES = ('stabilizer', 'resolver')
TIMING_PHASES = (
    'outer',
    'student_generation/stabilizer', 'student_generation/resolver',
    'student_context/stabilizer', 'student_forward/stabilizer', 'student_forward/resolver',
    'critic_pool_context/stabilizer', 'critic_pool_forward/stabilizer', 'critic_pool_forward/resolver',
    'input_generation/resolver',
    'teacher_critic_scoring/stabilizer', 'teacher_critic_scoring/resolver',
    'student_backward/stabilizer', 'student_backward/resolver',
    'diagnostics/stabilizer', 'diagnostics/resolver',
)
TIMING_ALIASES = {
    'student_forward': ('student_forward/stabilizer', 'student_forward/resolver'),
    'student_backward': ('student_backward/stabilizer', 'student_backward/resolver'),
    'teacher_critic_scoring': ('teacher_critic_scoring/stabilizer', 'teacher_critic_scoring/resolver'),
    'student_context': ('student_context/stabilizer',),
    'input_generation': ('input_generation/resolver',),
    'pool': ('critic_pool',),
    'critic_update': ('critic',),
}


def _weighted_total(values, weights):
    """One row weight applies to every scalar in that row, excluding padding."""
    weights = weights.detach().to(values).reshape(-1, *([1] * (values.ndim - 1)))
    total = (torch.where(weights != 0, values, 0) * weights).sum()
    count = weights.sum() * (values.numel() // values.shape[0])
    return total, count


@torch.no_grad()
def add_dmd_diagnostics(sums, kind, y, teacher, critic, target_velocity, weights):
    """Log prior DMD diagnostics without retaining graphs or consuming RNG.

    Sums reduces totals/counts across ranks before RMS is taken. Thus padded
    rows and unequal valid-row counts cannot distort global diagnostics.
    """
    if kind not in OBJECTIVES or y.ndim != 5 or not (
            y.shape == teacher.shape == critic.shape == target_velocity.shape):
        raise ValueError('DMD diagnostics require one objective and matching [B,C,F,H,W] tensors')
    if weights.shape != (y.shape[0],):
        raise ValueError('DMD diagnostic weights must contain one value per row')
    yt, vt, vf, target = (value.detach().float() for value in (y, teacher, critic, target_velocity))
    normalizer = (vt - target).abs().mean((1, 3, 4))  # Unclamped, as in the prior trainers.
    difference = (vt - vf).abs().mean((1, 3, 4))
    cosine = F.cosine_similarity(vt.flatten(1), vf.flatten(1), dim=1)
    velocity_mse = (vt - vf).square().flatten(1).mean(1)
    output_mse = yt.square().flatten(1).mean(1)
    values = {
        f'dmd_{kind}/normalizer': normalizer,
        f'dmd_{kind}/difference': difference,
        f'diagnostic/{kind}/teacher_critic_velocity_cosine': cosine,
        f'diagnostic/{kind}/teacher_critic_velocity_mse': velocity_mse,
        f'diagnostic/{kind}/output_mse': output_mse,
    }
    for name, value in values.items():
        sums.add(name, *_weighted_total(value, weights))


@torch.no_grad()
def add_resolver_history_dropout(sums, original_frame_indices, weights, *, target_start):
    """Count the actual two history slots for each of four packed targets.

    Five parent frames are shared as adjacent pairs: [0,1], [1,2], [2,3],
    [3,4]. This is the prior eight-slots-per-row metric, with tail padding
    excluded by the same weights used by the training loss.
    """
    if weights.shape != (original_frame_indices.shape[0],) or target_start < 2:
        raise ValueError('Resolver history diagnostic requires valid row weights and two preceding frames')
    invalid = original_frame_indices[:, target_start - 2:target_start + 3] < 0
    if invalid.shape[1] != 5:
        raise ValueError('Four resolver targets require five shared parent history frames')
    slots = torch.cat((invalid[:, :4], invalid[:, 1:]), dim=1).float()
    sums.add('data/resolver_history_slots_dropped', *_weighted_total(slots, weights))


def finalize_diagnostics(metrics, *, learning_rates, warming):
    """Convert globally reduced means to RMS and add exact scalar aliases."""
    result = dict(metrics)
    for kind in OBJECTIVES:
        for source, target in (('teacher_critic_velocity_mse', 'teacher_critic_velocity_rms'),
                               ('output_mse', 'output_rms')):
            key = f'diagnostic/{kind}/{source}'
            if key in result:
                result[f'diagnostic/{kind}/{target}'] = result.pop(key) ** .5
        output = f'diagnostic/{kind}/output_rms'
        if output in result:
            result[f'diagnostic/{kind}_output_rms'] = result[output]
        for role, old_prefix in (('critic', 'fm'), ('student', 'dmd')):
            key = f'{role}/{kind}'
            if key in result:
                result[f'{old_prefix}_{kind}/loss'] = result[key]
            for frame in range(4):
                key = f'{role}/{kind}/frame_{frame}'
                if key in result:
                    result[f'{old_prefix}_{kind}_frame/{frame}'] = result[key]
    for role, rate in learning_rates.items():
        result[f'{role}/lr_next'] = rate
    result['phase/critic_warmup'] = int(warming)
    result['phase/joint'] = int(not warming)
    return result


def add_timing_aliases(seconds):
    """Aggregate modes on each rank BEFORE the learner's MAX-rank reduction."""
    result = dict(seconds)
    for alias, names in TIMING_ALIASES.items():
        result[f'seconds/{alias}'] = sum(seconds[f'seconds/{name}'] for name in names)
    return result


def _training_metric_map():
    pairs = [
        ('train/round', 'rounds'), ('train/student_loss', 'student/loss'),
        ('train/critic_loss', 'critic/loss'), ('train/student_updates', 'student_updates'),
        ('train/critic_updates', 'critic_updates'), ('train/epoch_fraction', 'epoch_fraction'),
        ('train/student_lr_used', 'student/lr_used'), ('train/critic_lr_used', 'critic/lr_used'),
        ('train/seconds_per_round', 'seconds/wall'),
        ('progress/warmup_round', 'warmup_step'), ('progress/epoch_round', 'epoch_step'),
        ('progress/epoch_total_rounds', 'epoch_steps'), ('progress/student_rows_seen', 'student_rows_seen'),
        ('batch/valid_rows', 'global_valid_rows'), ('runtime/benchmark', 'benchmark'),
        ('dropout/resolver_history_slots_fraction', 'data/resolver_history_slots_dropped'),
        ('dropout/stabilizer_history_slots_fraction', 'data/stabilizer_history_slots_dropped'),
    ]
    for kind in OBJECTIVES:
        pairs += [(f'batch/{kind}_rows', f'global_objective_rows/{kind}'),
                  (f'batch/{kind}_targets', f'global_objective_targets/{kind}'),
                  (f'objective_weight/{kind}', f'objective_weights/{kind}')]
        for role in ('student', 'critic'):
            pairs.append((f'{kind}_{role}/loss', f'{role}/{kind}'))
            pairs.extend((f'{kind}_{role}/frame_{frame}', f'{role}/{kind}/frame_{frame}')
                         for frame in range(4))
        pairs.extend((f'{kind}_student/{metric}', f'dmd_{kind}/{metric}')
                     for metric in ('normalizer', 'difference'))
        pairs.extend((f'{kind}_diagnostics/{name}', f'diagnostic/{kind}/{source}') for name, source in (
            ('velocity_cosine', 'teacher_critic_velocity_cosine'),
            ('velocity_rms', 'teacher_critic_velocity_rms'), ('output_rms', 'output_rms')))
    for role in ('student', 'critic'):
        pairs += [(f'gradients/{role}_norm', f'{role}/grad_norm'),
                  (f'learning_rate/{role}_next', f'{role}/lr_next')]
    for phase in ('critic_warmup', 'joint'):
        pairs.append((f'phase/{phase}', f'phase/{phase}'))
    for name in ('peak_allocated_gib', 'peak_reserved_gib', 'allocated_start_gib',
                 'reserved_start_gib', 'allocated_end_gib', 'reserved_end_gib'):
        pairs.append((f'memory/{name}', f'memory/{name}'))
    for name in ('unique_graphs', 'new_graphs', 'graph_breaks', 'frames_total', 'frames_ok'):
        pairs.append((f'compile/{name}', f'compile/{name}'))
    for name in ('input_wait', 'gc', 'prepare', 'critic_pool', 'critic', 'critic_optimizer',
                 'student', 'student_optimizer', 'outer', 'student_forward',
                 'student_backward', 'teacher_critic_scoring'):
        pairs.append((f'timing/{name}', f'seconds/{name}'))
    for name in TIMING_PHASES:
        if '/' in name:
            phase, kind = name.split('/')
            pairs.append((f'timing_{kind}/{phase}', f'seconds/{name}'))
    mapping = {}
    for key, source in pairs:
        if key in mapping:
            raise ValueError(f'Online metric name collision: {key}')
        mapping[key] = source
    return mapping


TRAINING_METRIC_MAP = _training_metric_map()
# Local backwards-compatible aliases describe the same values. Sending them
# all online would create duplicate charts rather than additional information.
TRAINING_METRIC_ALIASES = {'train/round': 'rounds'}
for _kind in OBJECTIVES:
    TRAINING_METRIC_ALIASES[f'diagnostic/{_kind}_output_rms'] = f'diagnostic/{_kind}/output_rms'
    for _role, _prefix in (('critic', 'fm'), ('student', 'dmd')):
        TRAINING_METRIC_ALIASES[f'{_prefix}_{_kind}/loss'] = f'{_role}/{_kind}'
        for _frame in range(4):
            TRAINING_METRIC_ALIASES[f'{_prefix}_{_kind}_frame/{_frame}'] = f'{_role}/{_kind}/frame_{_frame}'
for _alias, _sources in TIMING_ALIASES.items():
    if len(_sources) == 1:
        TRAINING_METRIC_ALIASES[f'seconds/{_alias}'] = f'seconds/{_sources[0]}'
TRAINING_METADATA_FIELDS = {'phase'}  # Converted to explicit numeric phase flags.
TRAINING_DERIVED_FIELDS = ('throughput/unique_rows_per_second', 'throughput/objective_targets_per_second')


def project_training_metrics(raw):
    """Project known local scalars to broad online groups, without duplication.

    Missing historical fields stay absent. Throughput uses actual valid rows
    and the sum of the two objective target counts, never critic-pool reuse.
    Unknown numeric fields or conflicting aliases fail visibly so a newly
    added local metric cannot silently disappear from the online contract.
    This helper performs no tensor operations, RNG calls, or GPU activation.
    """
    flattened = {}

    def flatten(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(item, f'{prefix}/{key}' if prefix else str(key))
        else:
            if prefix in flattened:
                raise ValueError(f'Flattened training metric collision: {prefix}')
            flattened[prefix] = value

    flatten(raw)
    phase = flattened.get('phase')
    if phase is not None:
        if phase not in ('critic_warmup', 'joint'):
            raise ValueError(f'Unknown training phase: {phase}')
        for name in ('critic_warmup', 'joint'):
            flag = int(phase == name)
            if f'phase/{name}' in flattened and flattened[f'phase/{name}'] != flag:
                raise ValueError(f'Training phase indicator disagrees with phase: {name}')
            flattened[f'phase/{name}'] = flag
    accepted = set(TRAINING_METRIC_MAP.values()) | set(TRAINING_METRIC_ALIASES) | TRAINING_METADATA_FIELDS
    unknown = sorted(key for key, value in flattened.items()
                     if isinstance(value, (int, float)) and key not in accepted)
    if unknown:
        raise ValueError(f'Unmapped training scalar metrics: {unknown}')
    for alias, source in TRAINING_METRIC_ALIASES.items():
        if alias in flattened and source in flattened and flattened[alias] != flattened[source]:
            raise ValueError(f'Conflicting local metric alias: {alias} != {source}')
    result = {}
    for key, source in TRAINING_METRIC_MAP.items():
        value = flattened.get(source)
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[key] = int(value) if isinstance(value, bool) else value
    seconds = result.get('train/seconds_per_round', 0)
    if seconds > 0:
        rows = result.get('batch/valid_rows')
        if rows is not None:
            result[TRAINING_DERIVED_FIELDS[0]] = rows / seconds
        targets = [result.get(f'batch/{kind}_targets') for kind in OBJECTIVES]
        if all(value is not None for value in targets):
            result[TRAINING_DERIVED_FIELDS[1]] = sum(targets) / seconds
    return result

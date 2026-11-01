"""Fixed training scalar contract; all observations are detached weighted sums."""
import math

import torch
import torch.distributed as dist

METRIC_SCHEMA_VERSION = 'full-gt-training-v2'
OBJECTIVES = ('stabilizer', 'resolver')
BUCKETS = {
    'stabilizer': ('t_0.20_0.40', 't_0.40_0.60', 't_0.60_0.80', 't_0.80_1.00'),
    'resolver': ('t_0.00_0.05', 't_0.05_0.10', 't_0.10_0.15', 't_0.15_0.20'),
}
BUCKET_EDGES = {'stabilizer': (.2, .4, .6, .8, 1.), 'resolver': (0., .05, .1, .15, .2)}
VELOCITY_PROBES = (
    'student_rollout_rms', 'teacher_rms', 'critic_rms', 'critic_minus_teacher_mean',
    'critic_minus_teacher_variance', 'critic_minus_teacher_channel_bias_rms', 'teacher_critic_cosine',
)
TRAINING_METRIC_KEYS = frozenset(
    ['train/dmd_loss', 'train/critic_fm_loss', 'train/student_lr', 'train/critic_lr']
    + [f'train/{loss}_{kind}_loss' for loss in ('dmd', 'critic_fm') for kind in OBJECTIVES]
    + [f'{loss}_{kind}/timewise_{metric}/{bucket}'
       for loss, metrics in (('dmd', ('loss', 'normalizer')), ('critic_fm', ('endpoint_loss', 'velocity_loss')))
       for kind in OBJECTIVES for metric in metrics for bucket in BUCKETS[kind]]
    + [f'velocity_probes/{kind}/{name}' for kind in OBJECTIVES for name in VELOCITY_PROBES]
)


class TrainingMetrics:
    def __init__(self, device, channels=48):
        if type(channels) is not int or channels <= 0:
            raise ValueError('channels must be a positive integer')
        self.device, self.channels = torch.device(device), channels
        names = ['validation/invalid_weights', 'validation/invalid_times']
        names += ['train/student_lr', 'train/critic_lr']
        for kind in OBJECTIVES:
            names += [f'train/{loss}_{kind}_loss' for loss in ('dmd', 'critic_fm')]
            for bucket in BUCKETS[kind]:
                names += [f'{loss}_{kind}/timewise_{metric}/{bucket}'
                          for loss, metrics in (
                              ('dmd', ('loss', 'normalizer', 'raw_normalizer', 'normalizer_floor_hits')),
                              ('critic_fm', ('endpoint_loss', 'velocity_loss')),
                          ) for metric in metrics]
            prefix = f'velocity_probes/{kind}'
            names += [f'{prefix}/{role}_square' for role in ('student_rollout', 'teacher', 'critic')]
            names += [f'{prefix}/{name}' for name in (
                'delta', 'delta_square', 'teacher_critic_cosine', 'zero_norm_cosine_targets',
            )]
            names += [f'{prefix}/delta_channel_{channel}' for channel in range(channels)]
        self.names = tuple(names)
        self.index = {name: i for i, name in enumerate(names)}
        self.values = torch.zeros((len(names), 2), dtype=torch.float64, device=self.device)
        self.device = self.values.device

    def reset(self):
        self.values.zero_()

    def _add(self, name, total, count):
        self.values[self.index[name], 0].add_(total)
        self.values[self.index[name], 1].add_(count)

    def _mean(self, name, values, weights):
        values = torch.where(weights > 0, values, 0.)
        self._add(name, (values.double() * weights).sum(), weights.sum())

    def _weights(self, weights, shape):
        if weights.shape != shape[:1] or weights.device != self.device:
            raise ValueError('Expected weights [B] on the accumulator device')
        weights = weights.detach().double()
        valid = torch.isfinite(weights) & (weights >= 0)
        self._add('validation/invalid_weights', (~valid).double().sum(), weights.numel())
        return torch.where(valid, weights, 0.)[:, None].expand(shape)

    def _inputs(self, kind, frames, times, weights):
        if kind not in OBJECTIVES:
            raise ValueError(f'Unknown objective: {kind}')
        if frames.ndim != 2 or times.shape != frames.shape or weights.shape != frames.shape[:1]:
            raise ValueError('Expected frames/times [B,F] and weights [B]')
        if any(tensor.device != self.device for tensor in (frames, times, weights)):
            raise ValueError('Metric tensors must be on the accumulator device')
        if not times.is_floating_point():
            raise ValueError('Scoring times must be floating point')
        weights = self._weights(weights, frames.shape)
        times = times.detach()
        lower, *upper = BUCKET_EDGES[kind]
        in_range = (times > lower if kind == 'stabilizer' else times >= lower) & (times <= upper[-1])
        self._add('validation/invalid_times', ((weights > 0) & ~in_range).double().sum(), (weights > 0).sum())
        edges = times.new_tensor(upper[:-1])
        buckets = torch.bucketize(times.contiguous(), edges, right=False)
        return frames.detach(), weights, buckets

    def _velocity(self, value, frame_shape, weights):
        if (value.ndim != 5 or value.shape[0] != frame_shape[0] or value.shape[2] != frame_shape[1]
                or value.shape[1] != self.channels or value.device != self.device):
            raise ValueError('Expected velocity [B,C,F,H,W] with configured channel count')
        return torch.where((weights > 0)[:, None, :, None, None], value.detach().float(), 0.)

    def _bucket_means(self, kind, prefix, metric, values, weights, buckets):
        for index, bucket in enumerate(BUCKETS[kind]):
            self._mean(f'{prefix}_{kind}/timewise_{metric}/{bucket}', values, weights * (buckets == index))

    @torch.no_grad()
    def add_fm(self, kind, frames, times, velocity, target_velocity, weights):
        frames, weights, buckets = self._inputs(kind, frames, times, weights)
        velocity = self._velocity(velocity, frames.shape, weights)
        target = self._velocity(target_velocity, frames.shape, weights)
        if target.shape != velocity.shape:
            raise ValueError('Velocity and target shapes must match')
        raw_loss = (velocity - target).square().mean((1, 3, 4), dtype=torch.float64)
        self._mean(f'train/critic_fm_{kind}_loss', frames, weights)
        self._bucket_means(kind, 'critic_fm', 'endpoint_loss', frames, weights, buckets)
        self._bucket_means(kind, 'critic_fm', 'velocity_loss', raw_loss, weights, buckets)

    @torch.no_grad()
    def add_student_rollout(self, kind, velocity, weights):
        """Pool squared target velocities already computed during integration.

        Stabilizer observations cover each of its eight target evaluations;
        the resolver contributes its one target evaluation. Conditioning-slot
        outputs and detached critic-pool generation are excluded by the caller.
        Only sums/counts survive this call, never a velocity tensor or graph.
        """
        if kind not in OBJECTIVES:
            raise ValueError(f'Unknown objective: {kind}')
        if velocity.ndim != 5 or velocity.shape[2] != 1:
            raise ValueError('Expected one rollout target velocity [B,C,1,H,W]')
        shape = (velocity.shape[0], velocity.shape[2])
        weights = self._weights(weights, shape)
        velocity = self._velocity(velocity, shape, weights)
        square = velocity.square().sum((1, 3, 4), dtype=torch.float64)
        elements = velocity.shape[1] * velocity.shape[3] * velocity.shape[4]
        self._add(f'velocity_probes/{kind}/student_rollout_square',
                  (square * weights).sum(), weights.sum() * elements)

    @torch.no_grad()
    def add_dmd(self, kind, frames, times, teacher, critic, target_velocity, weights, epsilon):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('DMD epsilon must be finite and positive')
        frames, weights, buckets = self._inputs(kind, frames, times, weights)
        teacher, critic, target = [self._velocity(value, frames.shape, weights)
                                  for value in (teacher, critic, target_velocity)]
        if any(value.shape != teacher.shape for value in (critic, target)):
            raise ValueError('All same-input velocity shapes must match')
        # Match the FP32 reduction and clamp used by the actual DMD surrogate.
        raw = (teacher - target).abs().mean((1, 3, 4))
        normalizer = raw.clamp_min(epsilon)
        self._mean(f'train/dmd_{kind}_loss', frames, weights)
        for name, values in (('loss', frames), ('normalizer', normalizer),
                             ('raw_normalizer', raw), ('normalizer_floor_hits', (raw < epsilon).double())):
            self._bucket_means(kind, 'dmd', name, values, weights, buckets)

        prefix = f'velocity_probes/{kind}'
        elements = teacher.shape[1] * teacher.shape[3] * teacher.shape[4]
        count = weights.sum() * elements
        squares = []
        for role, velocity in (('teacher', teacher), ('critic', critic)):
            square = velocity.square().sum((1, 3, 4), dtype=torch.float64)
            self._add(f'{prefix}/{role}_square', (square * weights).sum(), count)
            squares.append(square)
        delta = critic - teacher
        self._add(f'{prefix}/delta', (delta.sum((1, 3, 4), dtype=torch.float64) * weights).sum(), count)
        self._add(f'{prefix}/delta_square', (delta.square().sum((1, 3, 4), dtype=torch.float64) * weights).sum(), count)
        channel_sums = (delta.sum((3, 4), dtype=torch.float64) * weights[:, None]).sum((0, 2))
        channel_count = weights.sum() * teacher.shape[3] * teacher.shape[4]
        start = self.index[f'{prefix}/delta_channel_0']
        self.values[start:start + self.channels, 0].add_(channel_sums)
        self.values[start:start + self.channels, 1].add_(channel_count)
        norm = (squares[0] * squares[1]).sqrt()
        valid = norm > 0
        cosine = (teacher * critic).sum((1, 3, 4), dtype=torch.float64) / torch.where(valid, norm, 1.)
        self._mean(f'{prefix}/teacher_critic_cosine', cosine, weights * valid)
        self._mean(f'{prefix}/zero_norm_cosine_targets', (~valid).double(), weights)

    @torch.no_grad()
    def add_lr(self, role, value):
        if role not in ('student', 'critic'):
            raise ValueError(f'Unknown optimizer role: {role}')
        if not math.isfinite(value) or value < 0:
            raise ValueError('Applied learning rates must be finite and nonnegative')
        self._add(f'train/{role}_lr', value, 1.)

    @torch.no_grad()
    def report(self, objective_weights, reduce=True):
        if (set(objective_weights) != set(OBJECTIVES) or
                any(not math.isfinite(value) or value <= 0 for value in objective_weights.values())):
            raise ValueError('Provide explicit finite positive coefficients for both objectives')
        values = self.values.clone()
        distributed = reduce and dist.is_available() and dist.is_initialized()
        if distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        rows = values.cpu().tolist()
        if any(not math.isfinite(value) for row in rows for value in row):
            raise FloatingPointError('Nonfinite training metric on a nonpadding observation')
        local = {name: {'sum': total, 'count': count} for name, (total, count) in zip(self.names, rows)}
        for name in ('invalid_weights', 'invalid_times'):
            if local[f'validation/{name}']['sum']:
                raise ValueError(f'Training metrics received {name.replace("_", " ")}')
        means = {name: row['sum'] / row['count'] for name, row in local.items() if row['count'] > 0}
        online = {key: value for key, value in means.items() if key in TRAINING_METRIC_KEYS}
        for loss in ('dmd', 'critic_fm'):
            if all(f'train/{loss}_{kind}_loss' in online for kind in OBJECTIVES):
                online[f'train/{loss}_loss'] = sum(objective_weights[kind] * online[f'train/{loss}_{kind}_loss']
                                                 for kind in OBJECTIVES)
        for kind in OBJECTIVES:
            prefix = f'velocity_probes/{kind}'
            for role in ('student_rollout', 'teacher', 'critic'):
                if f'{prefix}/{role}_square' in means:
                    online[f'{prefix}/{role}_rms'] = math.sqrt(means[f'{prefix}/{role}_square'])
            if f'{prefix}/delta' not in means:
                continue
            mean, second = means[f'{prefix}/delta'], means[f'{prefix}/delta_square']
            variance = second - mean * mean
            # Products are computed in FP32; only clamp its roundoff tolerance.
            if variance < -1e-6 * max(second, mean * mean, 1e-30):
                raise FloatingPointError('Negative critic-minus-teacher variance exceeds roundoff')
            online[f'{prefix}/critic_minus_teacher_mean'] = mean
            online[f'{prefix}/critic_minus_teacher_variance'] = max(0., variance)
            online[f'{prefix}/critic_minus_teacher_channel_bias_rms'] = math.sqrt(
                sum(means[f'{prefix}/delta_channel_{channel}'] ** 2 for channel in range(self.channels)) / self.channels)
        return online, local

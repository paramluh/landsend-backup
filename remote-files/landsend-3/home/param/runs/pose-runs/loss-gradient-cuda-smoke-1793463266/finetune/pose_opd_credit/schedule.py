"""Learning rates derived from the immutable full plan and resume cursor."""
import math


def resolve_schedule(config, total_updates):
    if type(total_updates) is not int or total_updates < 1:
        raise ValueError('Learning-rate schedule requires a positive update horizon')
    mode = config.learning_rate_schedule
    if mode == 'constant':
        return dict(mode=mode, total_updates=total_updates)
    if mode == 'amu_opd':
        # Amu speech_preservation.production.production_schedule: ceil5% and
        # ceil10%, 0.1 initial/final multipliers, stable middle.
        decay = math.ceil(.10 * total_updates)
        warmup = min(math.ceil(.05 * total_updates), total_updates - decay)
        if config.learning_rate_warmup_steps is not None:
            warmup = config.learning_rate_warmup_steps
        if warmup + decay > total_updates:
            raise ValueError('Warmup plus decay exceeds the full update plan')
        return dict(mode=mode, total_updates=total_updates, warmup_updates=warmup,
                    constant_updates=total_updates-warmup-decay, decay_updates=decay,
                    initial_lr_ratio=.1, minimum_lr_ratio=.1)
    if mode == 'source_linear':
        # Selected student export424:14 warmup then immediate linear decay to
        # zero at the horizon. Match native LambdaLR's completed-update index.
        warmup = max(1, round(total_updates * 14 / 424))
        if config.learning_rate_warmup_steps is not None:
            warmup = config.learning_rate_warmup_steps
        if not 0 < warmup < total_updates:
            raise ValueError('Source linear schedule needs 0 < warmup < total updates')
        return dict(mode=mode, total_updates=total_updates, warmup_updates=warmup,
                    constant_updates=0, decay_updates=total_updates-warmup,
                    initial_lr_ratio=0., minimum_lr_ratio=0.)
    raise ValueError(f'Unknown learning-rate schedule: {mode}')


def multiplier(schedule, completed_updates):
    """Multiplier for the next update; completed_updates is zero based."""
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError('Completed updates must be a nonnegative integer')
    if schedule['mode'] == 'constant':
        return 1.
    warmup = schedule['warmup_updates']
    if completed_updates < warmup:
        initial = schedule['initial_lr_ratio']
        return initial + (1. - initial) * completed_updates / warmup
    after_constant = completed_updates - warmup - schedule['constant_updates']
    if after_constant < 0:
        return 1.
    decay = schedule['decay_updates']
    if schedule['mode'] == 'source_linear':
        # Native source reaches zero AFTER the final optimizer update.
        fraction = min(1., after_constant / decay)
    else:
        # Amu's LAST applied decay update uses the minimum.
        fraction = 1. if decay == 1 else min(1., after_constant / (decay - 1))
    return 1. + (schedule['minimum_lr_ratio'] - 1.) * fraction


def apply_learning_rate(optimizer, peak, schedule, completed_updates):
    scale = multiplier(schedule, completed_updates)
    value = peak * scale
    for group in optimizer.param_groups:
        # Keep per-module peak ratios through warmup, plateau, cooldown and
        # resume. Legacy checkpoints without peak_lr use the global peak.
        group['lr'] = group.get('peak_lr', peak) * scale
    return value

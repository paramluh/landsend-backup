"""Reduce KL and detached dataset CE as sums/counts, including whole evaluations."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist

from .losses import paired_metric_namespaces


@torch.no_grad()
def collect(out, b, media, world, dev):
    """Collect globally summed statistics for one distributed batch.

    These values may be summed over arbitrarily sized batches before summarize.
    Dummy ranks and padded evaluation items contribute zero counts. Never average
    already normalized per-batch/rank metrics: text tokens and selected decoder
    frames have different counts in different conversations.
    """
    text = out['text']['stats']
    valid = text['valid']
    raw = [text['kl_sum'], text['full_kl_sum'], text['data_ce_sum'],
           text['data_weighted_ce_sum'], valid.sum()]
    for name in ('word', 'pad', 'nw', 'eob'):
        mask = b[name].bool() & valid
        raw.extend([(text['full_kl'] * mask).sum(), (text['data_ce'] * mask).sum(), mask.sum()])
    word = b['word'].bool() & valid
    raw.extend([(text['lexical_kl'] * word).sum(), (text['data_word_ce_norm'] * word).sum()])
    result = {'text': torch.stack([torch.as_tensor(x, device=dev).double() for x in raw])}
    for stream, stat, light in (('audio', out['stats_s'], media[0]), ('pose', out['stats_p'], media[1])):
        result[stream] = torch.stack([
            torch.cat([stat['kl_sum'].double().reshape(1), light['kl'].double()]),
            torch.cat([stat['data_ce_sum'].double().reshape(1), light['ce'].double()]),
            torch.cat([stat['count'].double().reshape(1), light['count'].double()]),
        ]).to(dev)
    if dist.is_initialized():
        if dist.get_world_size() != world:
            raise ValueError('Metric reduction world does not match the distributed process group')
        for value in result.values():
            dist.all_reduce(value)
    elif world != 1:
        raise ValueError('Multi-rank metrics require an initialized distributed process group')
    return result


@torch.no_grad()
def sum_raw(raws: Iterable[Mapping[str, torch.Tensor]]):
    """Sum batch statistics without changing the input values or denominators."""
    iterator = iter(raws)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError('Cannot summarize an evaluation with no batches') from exc
    result = {key: value.detach().clone() for key, value in first.items()}
    for batch in iterator:
        if set(batch) != set(result):
            raise ValueError('Raw metric groups differ between batches')
        for key, value in batch.items():
            if value.shape != result[key].shape:
                raise ValueError(f'Raw metric shape differs between batches: {key}')
            result[key].add_(value.detach())
    return result


@torch.no_grad()
def summarize(raw, cfg):
    """Normalize global sums once into KL and detached data-CE namespaces."""
    text = raw['text']
    if text.shape != (19,) or raw['audio'].shape != (3, 32) or raw['pose'].shape != (3, 32):
        raise ValueError('Raw metrics must contain text[19], audio[3,32] and pose[3,32]')
    denom = text[4].clamp_min(1)
    kl = {'text_loss': text[0] / denom}
    ce = {'text_loss': text[3] / denom}
    text_kl = {'full': text[1] / denom}
    text_ce = {'full': text[2] / denom}
    for i, name in enumerate(('word', 'pad', 'nw', 'eob')):
        offset = 5 + 3 * i
        count = text[offset + 2].clamp_min(1)
        text_kl[name] = text[offset] / count
        text_ce[name] = text[offset + 1] / count
    text_kl['word_conditional'] = text[-2] / text[7].clamp_min(1)
    text_ce['word_conditional'] = text[-1] / text[7].clamp_min(1)
    modality_weights = cfg['modality_weights']
    if set(modality_weights) != {'text', 'audio', 'pose'}:
        raise ValueError('Modality weights must specify text, audio and pose')
    kl['loss'] = modality_weights['text'] * kl['text_loss']
    ce['loss'] = modality_weights['text'] * ce['text_loss']
    books = []
    for stream, w0, w1 in (('audio', cfg['w_cb0'], cfg['w_cb']), ('pose', cfg['w_pcb0'], cfg['w_pcb'])):
        values = raw[stream]
        k = values[0] / values[2].clamp_min(1)
        c = values[1] / values[2].clamp_min(1)
        books.extend([k, c])
        kl[stream + '_loss'] = w0 * k[0] + w1 * k[1:].mean()
        ce[stream + '_loss'] = w0 * c[0] + w1 * c[1:].mean()
        kl['loss'] = kl['loss'] + modality_weights[stream] * kl[stream + '_loss']
        ce['loss'] = ce['loss'] + modality_weights[stream] * ce[stream + '_loss']
        lightkey = 'light' if stream == 'audio' else 'pose_light'
        kl[lightkey], ce[lightkey] = k[1:].mean(), c[1:].mean()
    metrics = paired_metric_namespaces(kl, ce, *books, text_kl=text_kl, text_ce=text_ce)
    return {key: float(value) for key, value in metrics.items()}

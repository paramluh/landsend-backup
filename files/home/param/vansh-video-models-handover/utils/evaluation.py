import torch


REF_KEYS = ('ref_latents', 'ref_keep')

def _gather(values, indices):
    B, C, _, H, W = values.shape
    gather = indices.clamp(min=0)[:, None, :, None, None].expand(B, C, indices.shape[1], H, W)
    return torch.gather(values, 2, gather) * (indices >= 0)[:, None, :, None, None]


def _source_noise(latents, original_frame_indices, source_ids, t):
    """Seeded per (source, frame) wherever t > 0, zero elsewhere."""
    noise = torch.zeros_like(latents)
    for batch_index, source_id in enumerate(source_ids):
        for slot in torch.nonzero(t[batch_index] > 0).flatten().tolist():
            frame = int(original_frame_indices[batch_index, slot])
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(17_123 + int(source_id) * 1_000_003 + frame)
            noise[batch_index, :, slot] = torch.randn(
                latents[batch_index, :, slot].shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
    return noise


def prepare_eval_batch(model, inputs, suite):
    """
    One full-frontier window per clip, the only layouts training and inference produce:
    window-0 = [sink | no history | frames 1..P], window-1 = [sink | frames 1..H | frames H+1..H+P].
    Noise levels t = model.eval_t(ofi).
    """
    H, P = model.num_history, model.num_present
    if suite not in ('window-0', 'window-1'):
        raise ValueError(f'Invalid eval suite: {suite!r}')
    sources = inputs['latents_low'].float()
    B = sources.shape[0]
    valid = inputs.get('valid', torch.ones(B, dtype=torch.bool))
    ofi = torch.full((B, 1 + H + P), -1, dtype=torch.long)
    ofi[:, 0] = 0
    present_start = 1
    if suite == 'window-1':
        ofi[:, 1:1 + H] = torch.arange(1, 1 + H)
        present_start = 1 + H
    ofi[:, 1 + H:] = torch.arange(present_start, present_start + P)
    latents = _gather(sources, ofi)
    t = model.eval_t(ofi)
    prepared = {
        'latents_low': latents,
        'pose_latents_low': _gather(inputs['pose_latents_low'].float(), ofi),
        'original_frame_indices': ofi,
        'present_valid': valid[:, None].repeat(1, P),
        'window_source': torch.arange(B),
        'codes': inputs['codes'],
        't': t,
        'noise': _source_noise(latents, ofi, inputs['source_id'], t),
    }
    for key in REF_KEYS:
        if key in inputs:
            prepared[key] = inputs[key]
    if 'llm_hiddens' in inputs:  # frames 1..H+P
        prepared['llm_hiddens'] = inputs['llm_hiddens'][:, present_start - 1:present_start - 1 + P]
    return model.prepare_inputs(prepared)


def _slots(buckets):
    return {int(name.split('-')[1]): value for name, value in buckets.items()}


@torch.no_grad()
def evaluate_suite(model, layered, loss_fn, eval_loader, suite, swap_refs=True):
    """
    Mean loss per present slot, accumulated through the model's slot buckets (same definition as loss_buckets/).
    With swap_refs and donor refs in the batch, each window is scored again with only its refs swapped -> second
    dict, else None.
    """
    model.reset_train_loss_buckets()
    real = model._train_loss_bucket_sums, model._train_loss_bucket_counts
    swapped = None
    for batch in eval_loader:
        features, labels = prepare_eval_batch(model, batch, suite)
        loss_fn(layered(features), labels)
        if swap_refs and 'donor_ref_latents' in batch:
            swapped = swapped or (torch.zeros_like(real[0]), torch.zeros_like(real[1]))
            model._train_loss_bucket_sums, model._train_loss_bucket_counts = swapped
            features, labels = prepare_eval_batch(model, {**batch, **{key: batch['donor_' + key] for key in REF_KEYS}}, suite)
            loss_fn(layered(features), labels)
            model._train_loss_bucket_sums, model._train_loss_bucket_counts = real
    buckets = _slots(model.get_train_loss_buckets())
    model.reset_train_loss_buckets()
    if swapped is None:
        return buckets, None
    model._train_loss_bucket_sums, model._train_loss_bucket_counts = swapped
    swapped_buckets = _slots(model.get_train_loss_buckets())
    model._train_loss_bucket_sums, model._train_loss_bucket_counts = real
    return buckets, swapped_buckets


def evaluate_suites(model, layered, loss_fn, eval_loader, swap_refs=True):
    """
    {(suite, slot): mean eval loss of present slot `slot` (1 = cleanest, P = newest) in a full-frontier window},
    plus (f'{suite}/swapped-refs', slot) when swap_refs and the loader yields donor refs.
    """
    layered.eval()
    try:
        results = {}
        for suite in ('window-0', 'window-1'):
            buckets, swapped = evaluate_suite(model, layered, loss_fn, eval_loader, suite, swap_refs)
            results.update({(suite, slot): value for slot, value in buckets.items()})
            if swapped is not None:
                results.update({(f'{suite}/swapped-refs', slot): value for slot, value in swapped.items()})
        return results
    finally:
        layered.train()

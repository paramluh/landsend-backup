"""Optional actor/replay depth comparison on identical final-frame roots."""
import time


def same_root_depth_probe(engine, rows, runtime):
    """Compare decoder implementations with exactly equal hv/zb and32 codes.

    Called after synchronized timing/readback. This changes decoder KV scratch
    only and never model weights. Only rows reaching the batch's last frame
    contribute; short rows' artificial continuation slots are excluded.
    """
    import numpy as np
    import torch
    started = time.perf_counter()
    last_frame = max(row['conversation_frames'] for row in rows) + 1
    eligible = [index for index, row in enumerate(rows) if row['conversation_frames'] + 1 == last_frame]
    result = {'frame': last_frame, 'instances': eligible,
              'interpretation': 'Identical tower roots, D33 roots, and actual32 sampled codebooks; isolates depth arithmetic from temporal drift.',
              'streams': {}}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for stream, decoder, hv_source, zb_source, device, code_buffer in (
            ('audio', engine.dec_s, engine.hv_s, None, engine.dl, engine.depB),
            ('pose', engine.dec_p, engine.hv_p[:engine.NI], engine.zb_p[:engine.NI], engine.dp, engine.pdepB)):
            with torch.cuda.device(device):
                hv = hv_source.detach().clone()
                zb = None if zb_source is None else zb_source.detach().clone()
                positions = engine.Tv[device][:engine.NI]
                global_codes = code_buffer[engine.arI[device], positions].detach().clone()
                codes = global_codes - torch.arange(32, device=device) * 2048
                if not bool(((codes >= 0) & (codes < 2048)).all()):
                    raise ValueError('Final-frame probe codebooks are invalid')
                flat = torch.arange(engine.NI, device=device)
                reference_x, _ = runtime._decoder_inputs(decoder.dec, hv, global_codes, flat, zb)
                x = torch.cat([decoder.slot0(hv, zb)[:, None], decoder.dec.cb0_emb(codes[:,0])[:,None].float()], 1)
                if not torch.equal(x, reference_x[:,:2]):
                    raise RuntimeError('Probe roots differ before the depth transformer')
                fast_logits = []
                for codebook in range(1,32):
                    fast_logits.append(decoder.head(decoder.step(x, codebook-1), codebook).float())
                    if codebook < 31:
                        x = decoder.emb(codes[:,codebook], codebook)
                fast_logits = torch.stack(fast_logits, 1)
                cos, sin = decoder.dec.model.rotary_emb(reference_x,
                    torch.arange(reference_x.shape[1], device=device)[None])
                hidden = reference_x
                for layer in decoder.dec.model.layers:
                    hidden = runtime._depth_layer(layer, hidden, cos, sin)
                hidden = runtime._depth_norm(decoder.dec.model.norm, hidden)
                reference_logits = torch.einsum('ndh,dvh->ndv', hidden[:,1:],
                    decoder.dec.heads.weight.view(31,2048,-1)).float()
                selected = torch.tensor(eligible, device=device)
                fast = fast_logits.index_select(0, selected).double()
                reference = reference_logits.index_select(0, selected).double()
                logp, logq = fast.log_softmax(-1), reference.log_softmax(-1)
                kl = (logp.exp()*(logp-logq)).sum(-1)
                reverse = (logq.exp()*(logq-logp)).sum(-1)
                delta = (fast-reference).abs()
                def metrics(index=None):
                    a = kl if index is None else kl[:,index]
                    b = reverse if index is None else reverse[:,index]
                    d = delta if index is None else delta[:,index]
                    return {'actor_to_replay_kl_mean': float(a.mean()), 'actor_to_replay_kl_max': float(a.max()),
                            'replay_to_actor_kl_mean': float(b.mean()), 'logit_mean_abs': float(d.mean()),
                            'logit_max_abs': float(d.max())}
                result['streams'][stream] = {'all31': metrics(),
                    'by_codebook': {str(book): metrics(book-1) for book in (1,14,31)},
                    'tower_root_rms': float(hv.float().square().mean().sqrt()),
                    'd33_root_rms': None if zb is None else float(zb.float().square().mean().sqrt()),
                    'slot0_rms': float(reference_x[:,0].square().mean().sqrt()),
                    'root_equal': True, 'input_dtype': str(reference_x.dtype)}
    for device in engine.devs:
        torch.cuda.synchronize(device)
    result['seconds'] = time.perf_counter() - started
    return result



"""One-step resolver DMD with ground-truth context for every target.

One row is [sink | six GT context frames | four target frames]. Resolver
student/teacher/critic passes share only GT clean histories. Method 2's frozen
stabilizer independently transports each target through its eight-step 2222
schedule; every other history/present input is freshly constructed from GT
and fixed per-frame noise. No generated target becomes another target's context.
"""

import torch

from distillation.bimodal_dmd import H, HR, P, SINK_SLOT, dmd_loss, fm_loss, noised

DATA_HISTORY = 6


def target_start(rows):
    """Index of the first of four target frames in a source row."""
    start = rows['latents'].shape[2] - P
    if start < 1 + HR:
        raise ValueError('Resolver rows require a sink and two preceding GT frames')
    return start


def resolve_gt_pass(net, rows, x_present, t_present, audio, audio_mask):
    """Four independent resolver windows, packed with five shared GT histories.

    x_present: [B,C,4,h,w], t_present: [B,4]. Returns the same velocity shape.
    GT frames target_start-2..target_start+2 supply ALL history, including the
    GT versions of targets 0..2; x_present never supplies a history key/value.
    """
    gt, pose = rows['latents'], rows['pose']
    start = target_start(rows)
    if x_present.shape != gt[:, :, start:].shape or t_present.shape != (gt.shape[0], P):
        raise ValueError('Resolver present/time shapes do not match four targets')
    frames = [(x_present[:, :, j], pose[:, :, start+j], t_present[:, j], HR+1, True)
              for j in range(P)]
    frames += [(gt[:, :, start-2+i], pose[:, :, start-2+i], 0., HR,
                rows['ofi'][:, start-2+i] >= 0)
               for i in range(P+1)]
    sink = 2*P+1
    frames.append((gt[:, :, 0], None, 0., SINK_SLOT, True))
    present = [[('f', sink), ('r', P+j, 1), ('r', P+j+1, 2), ('f', j)] for j in range(P)]
    history = [[('f', sink), ('f', P+i)] for i in range(P+1)]
    plan = [(0, present), (P, history), (sink, [[('f', sink)]])]
    velocity, _, _ = net.forward_pass(frames, plan, None, audio, audio_mask, P)
    return velocity


def gt_noised_input(rows, noise, t_start=.2):
    """Method 1: exact rectified-flow noising of each GT target at t_start."""
    gt = rows['latents'][:, :, target_start(rows):]
    if noise.shape != gt.shape:
        raise ValueError('Method-1 noise must have the four-target shape')
    return (1-t_start)*gt + t_start*noise


def one_step_resolve(net, rows, x02, tokens=None, t_start=.2):
    """One differentiable Euler step x_0 = x_.2 - .2*v(x_.2; GT history)."""
    if tokens is None:
        tokens = net.audio_tokens(rows['audio'])
    start = target_start(rows)
    audio = net.frame_audio(tokens, rows['ofi'][:, start:], rows['drop'])
    times = x02.new_full((x02.shape[0], P), t_start)
    mask = torch.ones(times.shape, dtype=torch.bool, device=x02.device)
    return x02 - t_start*resolve_gt_pass(net, rows, x02, times, audio, mask)


class GTConditionedStabilizer:
    """Frozen on-policy target generation with entirely GT temporal context.

    Four independent targets are flattened into the batch dimension. Every
    forward has the usual eight-frame stabilizer geometry. Future present
    slots are masked pads: the causal target cannot attend them. Histories
    are GT at history_t, and earlier present slots are GT noised at their
    appropriate band time. Fixed per-frame noise is reused across all passes.
    """

    def __init__(self, band_steps=(2, 2, 2, 2), history_t=.2):
        if len(band_steps) != P or len(set(band_steps)) != 1 or min(band_steps) < 1:
            raise ValueError('GT stabilizer requires four equally subdivided bands')
        if not 0 < history_t < 1:
            raise ValueError('history_t must lie strictly inside (0,1)')
        self.steps = int(band_steps[0])
        self.history_t = history_t
        self.width = (1-history_t)/P

    @torch.no_grad()
    def __call__(self, net, rows, tokens=None, noise=None):
        gt, pose = rows['latents'], rows['pose']
        start = target_start(rows)
        if start < 1+DATA_HISTORY:
            raise ValueError('GT stabilizer needs six preceding GT frames plus sink')
        if noise is None:
            noise = torch.randn_like(gt)
        if noise.shape != gt.shape:
            raise ValueError('Method-2 fixed noise must cover the complete source row')
        if tokens is None:
            tokens = net.audio_tokens(rows['audio'])
        B, C, _, height, width = gt.shape

        def targets(tensor, offset):
            selected = tensor[:, :, start+offset:start+offset+P]
            return selected.transpose(1, 2).reshape(B*P, C, height, width)

        def gt_at(offset, time):
            return (1-time)*targets(gt, offset) + time*targets(noise, offset)

        sink_latent = gt[:, :, 0].repeat_interleave(P, dim=0)
        x = targets(noise, 0).clone()
        zero = torch.zeros_like(x)
        # Identical to the ordinary stabilizer attention plan. Every query's
        # three left frames are GT except the query's own transported value.
        sink = H+P
        present_specs = [[('f', sink)] +
                         [('r', P+w-1, w) if w <= H else ('f', w-H-1)
                          for w in range(1+j, 1+H+j)] + [('f', j)] for j in range(P)]
        history_specs = [[('f', sink), ('f', P+w)] for w in range(H)]
        plan = [(0, present_specs), (P, history_specs), (sink, [[('f', sink)]])]
        for band in reversed(range(P)):
            # Each target f sits at present slot band; causal context covers
            # f-band-3..f-1. All four targets have independent window copies.
            history = [gt_at(-band-H+w, self.history_t) for w in range(H)]
            history_pose = [targets(pose, -band-H+w) for w in range(H)]
            positions = torch.arange(P, device=gt.device)[:, None] + start-band
            positions = positions + torch.arange(P, device=gt.device)[None, :]
            positions = positions.clamp(max=gt.shape[2]-1)
            audio = net.frame_audio(tokens, rows['ofi'][:, positions.flatten()], rows['drop'])
            audio = audio.reshape(B*P, P, *audio.shape[2:])
            mask = (torch.arange(P, device=gt.device) <= band)[None].expand(B*P, -1)
            for substep in range(self.steps):
                frames = []
                for slot in range(P):
                    time = self.history_t+self.width*(slot+1)-self.width*substep/self.steps
                    if slot > band:
                        frames.append((zero, None, 0., H+1+slot, False))
                    else:
                        value = x if slot == band else gt_at(slot-band, time)
                        frames.append((value, targets(pose, slot-band), time, H+1+slot, True))
                frames += [(history[w], history_pose[w], self.history_t, H, True) for w in range(H)]
                frames.append((sink_latent, None, 0., SINK_SLOT, True))
                velocity, _, _ = net.forward_pass(frames, plan, None, audio, mask, P)
                x = x - (self.width/self.steps)*velocity[:, :, band]
        return x.reshape(B, P, C, height, width).transpose(1, 2).contiguous()

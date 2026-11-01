"""Packed stabilizer followed by direct ten-step frozen-teacher resolution.

Model/cache/adapter math comes from the unchanged production inference modules.
Committed student x_t=.2 is passed unchanged to the teacher; no renoising draw.
Each worker gets an explicit CSV row, public clip ID and seed from its config.

Run ``python -m distillation.render_long CONFIG.yaml --row CSV_ROW``. Use
``--preflight`` to exercise five latent frames including saves and media decode.
The gallery config requires explicit clips, batch_size=1, wavefront=true,
renoising=false and the validated student/teacher schedules. A
prior_attempt_output_dir is optional; when supplied, a full run also checks its
first committed student frame against that prior run. Use a separate output
directory for each checkpoint and a distinct global clip ID per worker.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
import yaml

from inference.bounded_decode import decode_wan22_vae_bounded
from bimodal import sampler
from bimodal.audio_tables import resolver_tokens
from bimodal import graph_resolver
from bimodal.graph_resolver import GraphResolver
from distillation.latent_export import persist_latents
from utils.common import AUTOCAST_DTYPE


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--row', type=int, required=True)
    parser.add_argument('--trace-only', action='store_true', help='Run only four generated frames and no media decoding')
    parser.add_argument('--preflight', action='store_true', help='Four generated frames including all serialization/media operations')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get('batch_size') != 1 or not config.get('wavefront'):
        raise ValueError('Long inference requires batch1 packed wavefront')
    selected = next(item for item in config['clips'] if item['csv_row']==args.row)
    index, seed = selected['global_id'], selected['seed']
    if seed != 2026 + index:
        raise ValueError('Seed must match explicit approved global clip ID')
    stab, res = config['stabilising'], config['resolving']
    if not (stab['history_t']==.2 and stab['num_history']==3 and stab['num_present']==4 and stab['substeps']==2):
        raise ValueError('Student must retain the current four-frame2222 schedule and history')
    if not (res['t_start']==.2 and res['num_steps']==10 and res['num_history']==2 and res['num_present']==1):
        raise ValueError('Teacher must use direct ten uniform steps from.2 to0')
    if config.get('renoising') is not False:
        raise ValueError('Explicit renoising:false is required')
    checkpoint = Path(stab['checkpoint_path'])
    metadata = json.loads((checkpoint.parent/'metadata.json').read_text())
    if stab.get('lora') != metadata.get('lora'):
        raise ValueError('Raw student LoRA config must exactly match checkpoint metadata')
    rows = list(csv.DictReader(Path(config['csv']).open()))
    row = rows[args.row]
    if int(row['global_id']) != index or int(row['seed']) != seed:
        raise ValueError('Config and authoritative CSV clip identity/seed differ')
    output = Path(config['output_dir'])
    if args.trace_only: output = output.parent/f'{output.name}-trace'
    if args.preflight: output = output.parent/f'{output.name}-preflight'
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frame_limit = 5 if (args.trace_only or args.preflight) else config['max_latent_frames']
    clip = sampler.load_chunk(str(Path(config['root_dir'])/row['paths']), config['pose_file'],
                              frame_limit, config['mimi_file'])
    latents, pose, code_table, codes = clip
    minimum = 5 if (args.trace_only or args.preflight) else config['minimum_latent_frames']
    if latents.shape[1] < minimum:
        raise ValueError(f'Clip shortened to{latents.shape[1]} frames; require{minimum}')
    stabilising, resolving = sampler.build_modes(config, Path(sampler.__file__).parent, 'cuda')
    cached = sampler.WavefrontNet(stabilising, latents.shape[2:], stab['substeps'], 'cuda',
                                 config['compile'], 1, stab['history_t'])
    resolver = GraphResolver(resolving, num_steps=10)
    ts = torch.linspace(.2, 0, 11, device='cuda')
    poses, code_rows = pose[None].cuda(), codes[None].cuda()
    resolver_tables = [code_table.cuda()]
    audio_cache = {}
    resolved = torch.zeros_like(poses); resolved[:, :, 0] = latents[:, 0].cuda()
    trace = {'csv_row': args.row, 'global_id': index, 'seed': seed, 'path': row['paths'],
             'student_checkpoint_id': metadata['checkpoint_id'], 'teacher_checkpoint': res['checkpoint_path'],
             'renoising': False, 'resolver_steps': 10, 'schedule': ts.cpu().tolist(),
             'direct_first_input_equal': False, 'resolver_draws_random_noise': False,
             'resolver_backend': 'cuda_graph',
             'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'graph_source_sha256': hashlib.sha256(Path(graph_resolver.__file__).read_bytes()).hexdigest()}
    prior_trace = None
    if config.get('prior_attempt_output_dir'):
        prior_trace_path = Path(config['prior_attempt_output_dir'])/f'direct-resolver-trace-{index}.json'
        prior_trace = json.loads(prior_trace_path.read_text())
        if prior_trace['student_checkpoint_id'] != metadata['checkpoint_id'] or prior_trace['seed'] != seed:
            raise ValueError('Prior execution checkpoint/seed identity differs')
        trace.update(prior_attempt_trace=str(prior_trace_path),
                     prior_attempt_trace_sha256=hashlib.sha256(prior_trace_path.read_bytes()).hexdigest())
    current = {}
    original_velocity = resolver.velocity

    def checked_velocity(x, *rest, **kwargs):
        if not trace['direct_first_input_equal']:
            expected = current['committed']
            assert torch.equal(x, expected), 'First teacher input differs from committed studentx.2'
            trace.update(direct_first_input_equal=True, student_commit_sha256=tensor_sha(expected),
                         teacher_first_input_sha256=tensor_sha(x), same_input_storage=x.data_ptr()==current['pointer'])
        return original_velocity(x, *rest, **kwargs)

    resolver.velocity = checked_velocity

    def resolve(frame, committed):
        tokens = resolver_tokens(resolver, resolver_tables, code_rows, frame, audio_cache)
        if frame==1:
            if prior_trace is not None and not (args.preflight or args.trace_only):
                assert tensor_sha(committed) == prior_trace['student_commit_sha256'], 'Retry first student commit differs from original execution'
                trace['first_student_commit_matches_prior_attempt'] = True
            current.update(committed=committed.clone(), pointer=committed.data_ptr())
            rng = torch.cuda.get_rng_state()
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            # No scaling, epsilon draw, or addition: exact student tensor enters teacher.
            resolver.frame(resolved, poses, code_rows, frame, committed, ts, tokens)
        if frame==1:
            assert torch.equal(committed, current['committed']), 'Teacher mutated committed student tensor'
            assert torch.equal(torch.cuda.get_rng_state(), rng), 'Teacher advanced CUDA RNG unexpectedly'
            trace['student_commit_unchanged'] = True
            trace['cuda_rng_unchanged_during_first_resolve'] = True
            current.clear()

    if not args.trace_only:
        audio = stabilising.mimi.decode(codes[None, :, :2*latents.shape[1]].cuda()).audio_values[0,0].float().cpu().numpy()
        if config.get('decode_ground_truth', True):
            decode_wan22_vae_bounded(stabilising.vae, latents[None], output/f'samples_{index}.mp4', 25, audio, 'long')
    generated = sampler.generate_wavefront(cached, [clip], [seed], 'cuda', config['cuda_graph'], on_commit=resolve)[0]
    assert trace['direct_first_input_equal']
    trace.update(complete=True, latent_frames=latents.shape[1], generated_frames=latents.shape[1]-1,
                 decoded_video_frames=4*latents.shape[1]-3, trace_only=args.trace_only, preflight=args.preflight,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    (output/f'direct-resolver-trace-{index}.json').write_text(json.dumps(trace,indent=2)+'\n')
    if not args.trace_only:
        # Persist BOTH model outputs before decoding either output. Contiguous CPU
        # packing is mandatory: the resolved view can inherit non-contiguous strides.
        outputs = [(f'sample_{index}', resolved[0]), (f'sample_{index}_stabilised', generated)]
        latent_receipts = persist_latents(output, outputs, verify=args.preflight)
        trace['latent_files'] = latent_receipts
        (output/f'direct-resolver-trace-{index}.json').write_text(json.dumps(trace,indent=2)+'\n')
        for name, values in outputs:
            decode_wan22_vae_bounded(stabilising.vae, values[None], output/f'{name}.mp4',25,audio,'long')
        if config.get('side_by_side', True):
            sampler.side_by_side(output/f'samples_{index}.mp4',output/f'sample_{index}.mp4',output/f'sample_{index}_sxs.mp4')
        shutil.copy2(config_path,output/f'config-{index}.yaml')
    trace['elapsed_seconds'] = time.monotonic()-started
    (output/f'worker-complete-{index}.json').write_text(json.dumps(trace,indent=2)+'\n')
    print(json.dumps(trace),flush=True)


if __name__=='__main__': main()

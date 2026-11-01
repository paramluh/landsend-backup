"""Compose the fixed checkpoint evaluation and retain its local gallery entry."""
import json
import hashlib
import os
from pathlib import Path

import torch

from distillation.checkpoints import file_sha256, write_json_atomic
from distillation.endpoint_sensitivity import run_endpoint_sensitivity, render_endpoint_sensitivity
from distillation.rollout_evaluation import evaluate_rollout, append_local_gallery
from distillation.training_tracking import EVALUATION_SCHEMA_VERSION
from distillation.evaluation_identity import evaluation_identity_fields


def _artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def _hash_tensor(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_artifacts(result):
    for artifact in result['artifacts'].values():
        if _artifact(artifact['path']) != artifact:
            raise ValueError('Saved evaluation artifact failed integrity verification')


def evaluate_checkpoint(net, rows, raw, audio, config, directory, step, weights, run_dir, wandb_url, *, identity=None):
    directory, run_dir = Path(directory), Path(run_dir)
    fields = evaluation_identity_fields(identity if identity is not None else
        json.loads((directory / 'metadata.json').read_text()))
    identity_path = None
    if identity is not None:
        if fields.get('evaluation_kind') != 'in_memory' or fields['rounds'] != step:
            raise ValueError('In-memory evaluation identity does not match the training round')
        directory.mkdir(parents=True, exist_ok=True)
        identity_path = directory / 'evaluation.identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Saved in-memory evaluation identity differs')
        if not identity_path.exists():
            write_json_atomic(identity_path, identity)
    artifact_id = fields.get('evaluation_id') or fields['checkpoint_id']
    # Weight retention deliberately permits only flat checkpoint directories.
    output = run_dir / 'evaluations' / artifact_id / weights
    output.mkdir(parents=True, exist_ok=True)
    completed = output / 'evaluation.json'
    report = {'weights': weights, 'file': os.path.relpath(completed, directory)}
    source = {'chunk_index': int(raw['chunk_index'][0]), 'person_index': int(raw['person_index'][0]),
              'source_frame_indices': raw['source_frame_indices'][0].tolist(),
              'source_clip_index': int(raw['epoch_window_id'][0]),
              'paths': raw.get('source_paths', {}),
              'csv': str(Path(config['clipnames_root']) / config['train_csv']),
              'bank': config['refs']['bank']}
    protocol = {'schema': EVALUATION_SCHEMA_VERSION, 'source': source, 'rollout': config['rollout_eval'],
                'band_steps': config['band_steps'], 'resolver_steps': config['resolver_steps'],
                'sensitivity_seeds': config['sensitivity_seeds'], 'sensitivity_rms': config['sensitivity_rms'],
                'sensitivity_validate_half_rms': config['sensitivity_validate_half_rms'],
                'inputs': {key: _hash_tensor(rows[key]) for key in
                           ('latents', 'pose', 'ofi', 'audio', 'refs', 'ref_keep', 'drop')}}
    protocol_path = run_dir / 'evaluation-protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Evaluation inputs/protocol changed within this run')
    else:
        write_json_atomic(protocol_path, protocol)
    if completed.exists():
        result = json.loads(completed.read_text())
        if (evaluation_identity_fields(result) != fields or
                (result['checkpoint_step'], result['schema'], result['weights']) != (
                step, EVALUATION_SCHEMA_VERSION, weights)):
            raise ValueError('Saved evaluation does not match the requested checkpoint/protocol')
        _verify_artifacts(result)
        return report
    options = {**config['rollout_eval'], 'run_id': run_dir.name, **fields,
               'weights': weights, 'source': source}
    rollout_path = output / 'rollout.json'
    if rollout_path.exists():
        result = json.loads(rollout_path.read_text())
        if (evaluation_identity_fields(result) != fields or
                (result['weights'], result['source']) != (weights, source)):
            raise ValueError('Saved rollout does not match this evaluation')
        _verify_artifacts(result)
    else:
        result = evaluate_rollout(net, rows, {**config, 'rollout_eval': options}, output, audio_np=audio)
    sensitivity_path = output / 'sensitivity.json'
    if sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text())
    else:
        sensitivity = run_endpoint_sensitivity(net, rows, config)
        write_json_atomic(sensitivity_path, sensitivity)
    figures = render_endpoint_sensitivity(sensitivity, output)
    result['artifacts'].update({name: _artifact(path) for name, path in figures.items()})
    result['artifacts']['sensitivity_measurements'] = _artifact(sensitivity_path)
    result['artifacts']['evaluation_protocol'] = _artifact(protocol_path)
    if identity_path is not None:
        result['artifacts']['evaluation_identity'] = _artifact(identity_path)
        result['evaluation_identity'] = dict(identity)
    result.update(fields)
    result.update(schema=EVALUATION_SCHEMA_VERSION, checkpoint_step=step, wandb_url=wandb_url)
    result['scalars'] = {key: value for key, value in result['scalars'].items() if value is not None}
    result.pop('manifest_path', None)
    result['gallery'] = append_local_gallery(result, run_dir / 'gallery')
    write_json_atomic(completed, result)
    return report

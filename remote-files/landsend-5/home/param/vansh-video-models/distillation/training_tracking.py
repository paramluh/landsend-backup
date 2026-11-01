"""Fixed training/evaluation W&B contract for full-GT endpoint training."""
import json
import math
from pathlib import Path
import uuid

from distillation.checkpoints import file_sha256, write_json_atomic
from distillation.training_metrics import METRIC_SCHEMA_VERSION, TRAINING_METRIC_KEYS
from distillation.evaluation_identity import evaluation_identity_fields

EVALUATION_SCHEMA_VERSION = 'endpoint-context-and-student-ar-30s-v1'
DRIFT_SCALARS = ('background_color_mean', 'background_color_tail',
                 'background_texture_mean', 'background_texture_tail')
FIGURES = {
    'stabilizer_context_gain': 'sensitivity', 'resolver_context_gain': 'sensitivity',
    'background_curves': 'drift', 'gt_l2_curve': 'rollout',
}


class TrainingTracking:
    def __init__(self, run, sdk, directory, receipt):
        self.run, self.sdk = run, sdk
        self.directory, self.receipt = Path(directory), receipt

    def _receipt(self):
        write_json_atomic(self.directory / 'wandb.json', self.receipt)

    def training(self, record):
        step = int(record['rounds'])
        payload = dict(record['training_metrics'])
        if set(payload) - TRAINING_METRIC_KEYS:
            raise ValueError(f'Unexpected training keys: {sorted(set(payload) - TRAINING_METRIC_KEYS)}')
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in payload.values()):
            raise ValueError('Training observations must be finite scalars')
        if step <= self.receipt.get('last_train_step', -1):
            raise ValueError('Training metrics must be published once per completed round')
        self.run.log({'train/step': step, **payload}, commit=True)
        self.receipt['last_train_step'] = step
        self._receipt()

    def evaluation(self, directory, counters, *, startup=False):
        directory = Path(directory)
        completion = json.loads((directory / 'evaluation.complete.json').read_text())
        identity = evaluation_identity_fields(completion)
        if identity.get('evaluation_kind') == 'in_memory' and (
                identity['rounds'] != counters['rounds'] or
                identity['student_updates'] != counters.get('student_updates')):
            raise ValueError('In-memory evaluation counters differ from completed training state')
        payload = {'eval/checkpoint_step': int(counters['rounds'])}
        files = []
        labels = set()
        if not completion['reports']:
            raise ValueError('Completed evaluation requires at least one weight variant')
        for report in completion['reports']:
            manifest_path = directory / report['file']
            manifest = json.loads(manifest_path.read_text())
            label = manifest['weights']
            if (label not in ('raw', 'ema') or label != report['weights'] or label in labels
                    or evaluation_identity_fields(manifest) != identity):
                raise ValueError('Evaluation weights/checkpoint identity mismatch')
            labels.add(label)
            if manifest['checkpoint_step'] != counters['rounds'] or manifest['status'] != 'complete':
                raise ValueError('Evaluation must match the completed checkpoint step')
            if set(manifest['scalars']) - set(DRIFT_SCALARS):
                raise ValueError('Unexpected drift scalar keys')
            for name, value in manifest['scalars'].items():
                if not math.isfinite(value):
                    raise ValueError('Missing drift observations must be omitted, not NaN')
                payload[f'eval/drift/{label}/{name}'] = value
            caption = (f"{label} · student update {identity['student_updates']} · in-memory {identity['evaluation_id']}"
                       if identity.get('evaluation_kind') == 'in_memory' else
                       f"{label} · checkpoint {counters['rounds']} · {manifest['checkpoint_id']}")
            artifacts = manifest['artifacts']
            for name, record in artifacts.items():
                path = Path(record['path'])
                if not path.is_absolute():
                    path = manifest_path.parent / path
                if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
                    raise ValueError(f'Evaluation artifact failed integrity check: {path}')
                files.append(path)
                if name in FIGURES:
                    payload[f'eval/{FIGURES[name]}/{label}/{name}'] = self.sdk.Image(str(path), caption=caption)
                elif name in ('generated_video', 'gt_video'):
                    payload[f'eval/rollout/{label}/{name}'] = self.sdk.Video(str(path), caption=caption, format='mp4')
            if not set(FIGURES) <= set(artifacts) or not {'generated_video', 'gt_video'} <= set(artifacts):
                raise ValueError('Completed evaluation is missing required plots or videos')
            files.append(manifest_path)
        # SDK steps are independent of checkpoint steps, including delayed eval.
        self.run.log(payload, commit=True)
        for path in files:
            self.run.save(str(path), base_path=str(self.directory), policy='now', glob=False)
        self.receipt['last_eval_checkpoint'] = completion['checkpoint_id']
        self.receipt['last_eval_id'] = identity.get('evaluation_id') or identity['checkpoint_id']
        self.receipt['last_eval_kind'] = identity.get('evaluation_kind', 'checkpoint')
        self.receipt['last_eval_step'] = int(counters['rounds'])
        self._receipt()
        return dict(self.receipt)

    def finish(self):
        self.run.finish()


def start_training_tracking(config, run_dir, counters, *, epoch_steps, world_size,
                            resume_checkpoint=None, wandb_module=None):
    settings = config['wandb']
    if settings['enabled'] is not True or settings['mode'] != 'online':
        raise ValueError('Production requires online W&B tracking')
    if wandb_module is None:
        import wandb as wandb_module
    directory = Path(run_dir).absolute()
    receipt_path = directory / 'wandb.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    existing_id, parent, last_train_step = None, None, -1
    if previous:
        if (previous['entity'], previous['project']) != (settings['entity'], settings['project']):
            raise ValueError('Existing W&B identity differs from configuration')
        remote = wandb_module.Api(timeout=15).run(f"{previous['entity']}/{previous['project']}/{previous['id']}")
        remote_step = max(remote.summary.get('train/step', -1), remote.summary.get('eval/checkpoint_step', -1),
                          previous.get('last_train_step', -1))
        if (remote_step > counters['rounds'] or previous.get('metric_schema') != METRIC_SCHEMA_VERSION
                or previous.get('evaluation_schema') != EVALUATION_SCHEMA_VERSION):
            parent = {'id': previous['id'], 'url': previous['url'], 'checkpoint_step': counters['rounds']}
        else:
            existing_id, parent = previous['id'], previous.get('parent')
            last_train_step = previous.get('last_train_step', -1)
    run = wandb_module.init(entity=settings['entity'], project=settings['project'], name=settings['run_name'],
        id=existing_id or uuid.uuid4().hex[:12], resume='must' if existing_id else 'never', mode='online',
        force=True, save_code=False, dir=str(directory), settings={'init_timeout': 90, 'console': 'off',
        'x_disable_stats': True, 'x_disable_meta': True}, config={**config, 'epoch_steps': epoch_steps,
        'world_size': world_size, 'metric_schema': METRIC_SCHEMA_VERSION,
        'evaluation_schema': EVALUATION_SCHEMA_VERSION, 'tracking_parent': parent,
        'tracking_resume_checkpoint': str(resume_checkpoint) if resume_checkpoint else None})
    try:
        if run is None or run.settings.mode != 'online' or run.offline or not run.url:
            raise RuntimeError('W&B did not establish an online run')
        run.define_metric('train/step', hidden=True)
        run.define_metric('eval/checkpoint_step', hidden=True)
        for key in sorted(TRAINING_METRIC_KEYS):
            run.define_metric(key, step_metric='train/step', step_sync=False)
        run.define_metric('eval/*', step_metric='eval/checkpoint_step', step_sync=False)
        receipt = {'id': run.id, 'url': run.url, 'entity': run.entity, 'project': run.project,
                   'metric_schema': METRIC_SCHEMA_VERSION, 'evaluation_schema': EVALUATION_SCHEMA_VERSION,
                   'mode': 'online', 'parent': parent, 'last_train_step': last_train_step,
                   'resume_checkpoint': str(resume_checkpoint) if resume_checkpoint else None}
        tracker = TrainingTracking(run, wandb_module, directory, receipt)
        tracker._receipt()
        if resume_checkpoint and (Path(resume_checkpoint) / 'evaluation.complete.json').exists():
            tracker.evaluation(resume_checkpoint, counters, startup=True)
        return tracker
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise

"""Online joint metrics and bounded sensitivity JSON; never uploads model weights."""
import json
import math
from pathlib import Path
import time
import uuid

from distillation.joint_metrics import project_training_metrics
from distillation.joint_sensitivity_tracking import (GROUPS, checkpoint_metrics,
    sensitivity_metrics, stage_checkpoint_reports, verify_report_files)


def numeric_metrics(value, prefix=''):
    """Flatten nested scalar metrics without passing tensors, files, or media."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f'{prefix}/{key}' if prefix else str(key)
            if key == 'status' and isinstance(item, str):
                result[f'{prefix}/status_ok'] = int(item == 'ok')
                result[f'{prefix}/status_skipped'] = int(item.startswith('skipped'))
            else:
                result.update(numeric_metrics(item, name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(numeric_metrics(item, f'{prefix}/{index}'))
    elif isinstance(value, (int, float)) and math.isfinite(value):
        result[prefix] = int(value) if isinstance(value, bool) else value
    return result


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class OnlineTracking:
    def __init__(self, run, api, directory, receipt, verify_timeout=60):
        self.run, self.api = run, api
        self.directory, self.receipt = Path(directory), receipt
        self.verify_timeout = verify_timeout

    def training(self, metrics):
        step = int(metrics['rounds'])
        if self.run.step > step:
            raise ValueError('W&B history is ahead of training; refusing to discard replayed metrics')
        payload = project_training_metrics(metrics)
        payload['train/round'] = step
        # Keep this round open so its later sensitivity results share its step.
        self.run.log(payload, step=step, commit=False)

    def publish(self, payload, counters, *, startup=False, report_files=()):
        step = int(counters['rounds'])
        payload = {**payload, 'train/round': step, 'train/student_updates': counters['student_updates'],
                   'train/critic_updates': counters['critic_updates'], 'tracking/online': 1}
        token = uuid.uuid4().hex
        payload['tracking/publish_token'] = token
        # A CPU preflight may already have committed the resume checkpoint's
        # row. Updating its summary must not consume the next training step.
        if self.run.step <= step:
            self.run.log(payload, step=step, commit=True)
        elif not startup:
            raise ValueError('W&B history is ahead of the evaluated checkpoint')
        # A failed/skipped site has no new amplitude scalar. Remove only its
        # compact latest-summary value so a preceding checkpoint cannot look
        # current. Historical rows and the original detailed namespace remain.
        active_weights = [label for label in ('raw', 'ema')
                          if f'sensitivity_overview/{label}_status_ok' in payload]
        prefixes = tuple(f'{group}/{label}_' for group in GROUPS[:3] for label in active_weights)
        cleared = [key for key in self.run.summary.keys() if prefixes and key.startswith(prefixes)
                   and key not in payload]
        for key in cleared:
            del self.run.summary[key]
        self.run.summary.update(payload)
        deadline = time.monotonic() + self.verify_timeout
        required = {key: value for key, value in payload.items()
                    if key in ('tracking/publish_token', 'tracking/online', 'train/round',
                               'train/student_updates', 'train/critic_updates', 'train/critic_loss', 'train/student_loss')
                    or key.startswith(tuple(group + '/' for group in GROUPS))}
        verified_files = []
        while True:
            self.api.flush()
            remote = self.api.run(f'{self.run.entity}/{self.run.project}/{self.run.id}')
            summary = dict(remote.summary)
            scalars_visible = all(summary.get(key) == value or (isinstance(value, (int, float))
                    and isinstance(summary.get(key), (int, float))
                    and math.isclose(summary[key], value, rel_tol=1e-6, abs_tol=1e-10))
                   for key, value in required.items())
            if scalars_visible and all(key not in summary for key in cleared):
                verified_files = verify_report_files(remote, report_files, api=self.api) if report_files else []
                if verified_files is not None:
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError('W&B metrics or sensitivity JSON were not verified through the remote API before timeout')
            time.sleep(2)
        self.receipt.update({'online_metrics_verified': True, 'verified_round': step,
                             'verified_student_updates': counters['student_updates'],
                             'verified_sensitivity_metrics': sum(key.startswith(tuple(group + '/' for group in GROUPS)) for key in required),
                             'sensitivity_report_files': verified_files,
                             'sensitivity_report_uploads_verified': bool(verified_files),
                             'cleared_stale_sensitivity_summary_keys': cleared,
                             'publish_token': token})
        _write_json(self.directory / 'wandb.json', self.receipt)
        return dict(self.receipt)

    def _stage_reports(self, directory):
        records = stage_checkpoint_reports(directory, self.directory)
        for record in records:
            self.run.save(record['path'], base_path=str(self.directory.absolute()), policy='now', glob=False)
        return records

    def evaluation(self, directory, counters, *, startup=False):
        return self.publish(checkpoint_metrics(directory), counters, startup=startup,
                            report_files=self._stage_reports(directory))

    def finish(self):
        self.run.finish()


def start_online_tracking(config, run_dir, counters, *, epoch_steps, world_size,
                          resume_checkpoint=None, wandb_module=None, verify_timeout=60):
    """Rank-zero entry point, also usable for a CPU-only real-run preflight.

    Only metrics at the resumed checkpoint are copied from local history. Later
    abandoned metrics remain in metrics.jsonl and never masquerade as replay.
    """
    settings = config.get('wandb', {})
    if settings.get('enabled') is not True or settings.get('mode') != 'online':
        raise ValueError('Production requires wandb.enabled=true and wandb.mode=online')
    if wandb_module is None:
        import wandb as wandb_module
    directory = Path(run_dir)
    receipt_path = directory / 'wandb.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    api = wandb_module.Api(timeout=15)
    parent = None
    existing_id = None
    if previous:
        if previous.get('project') != settings['project'] or previous.get('entity') != settings['entity']:
            raise ValueError('Existing W&B identity differs from configured entity/project')
        remote = api.run(f'{previous["entity"]}/{previous["project"]}/{previous["id"]}')
        if remote.lastHistoryStep > counters['rounds']:
            parent = {'id': previous['id'], 'url': previous['url'],
                      'last_history_step': remote.lastHistoryStep,
                      'reason': 'resume_checkpoint_precedes_remote_history'}
        else:
            existing_id = previous['id']
            parent = previous.get('parent')
    run = wandb_module.init(entity=settings['entity'], project=settings['project'],
        name=settings['run_name'], id=existing_id or uuid.uuid4().hex[:12],
        resume='must' if existing_id else 'never', mode='online', force=True, save_code=False,
        config={**config, 'epoch_steps': epoch_steps, 'world_size': world_size,
                'tracking_resume_checkpoint': str(resume_checkpoint) if resume_checkpoint else None,
                'tracking_parent': parent}, dir=str(directory),
        settings={'init_timeout': 90, 'console': 'off'})
    try:
        if run is None or run.settings.mode != 'online' or run.offline or not run.url:
            raise RuntimeError('W&B did not establish an online run')
        run.define_metric('train/round')
        run.define_metric('*', step_metric='train/round')
        run.define_metric('sensitivity_overview/checkpoint_round')
        for group in GROUPS:
            run.define_metric(group + '/*', step_metric='sensitivity_overview/checkpoint_round')
        receipt = {'url': run.url, 'id': run.id, 'entity': run.entity, 'project': run.project,
                   'mode': 'online', 'online_metrics_verified': False,
                   'parent': parent, 'resume_checkpoint': str(resume_checkpoint) if resume_checkpoint else None,
                   'checkpoint_artifact_uploads': False, 'sensitivity_json_uploads': True}
        tracker = OnlineTracking(run, api, directory, receipt, verify_timeout)
        _write_json(receipt_path, receipt)
        payload = {}
        history = directory / 'metrics.jsonl'
        if history.exists():
            for line in history.read_text().splitlines():
                row = json.loads(line)
                if row.get('rounds') == counters['rounds']:
                    payload = project_training_metrics(row)
        report_files = []
        if resume_checkpoint and (Path(resume_checkpoint) / 'evaluation.complete.json').exists():
            payload.update(checkpoint_metrics(resume_checkpoint))
            report_files = tracker._stage_reports(resume_checkpoint)
        tracker.publish(payload, counters, startup=True, report_files=report_files)
        return tracker
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise

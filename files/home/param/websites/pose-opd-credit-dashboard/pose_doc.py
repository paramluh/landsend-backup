#!/usr/bin/env python3
"""Read pose OPD logs without importing or changing the trainer.

The timestamped rank-zero ``update`` events contain the same reduced metrics as
metrics.jsonl and are preferred when both streams are present. No scheduler,
training process, checkpoint writer, or network service is invoked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


CORE_KEYS = (
    "loss/direct", "loss/text", "loss/audio", "loss/pose", "gradient_norm",
    "policy/logprob_mean_abs_difference", "policy/logprob_max_abs_difference",
    "train/step_time", "credit/absolute_sum", "learning_rate",
)
MORE_PREFIXES = ("loss/", "credit/", "policy/", "kl/", "grad_norm/",
                 "logit_grad_norm/", "logit_grad_alignment/", "logit_grad_credit_direct_ratio/", "logit_grad_dot/",
                 "loss_gradient_diagnostics_", "conversation/")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value, *, minimum=0):
    return int(value) if number(value) and value == int(value) and value >= minimum else None


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def timestamp(row):
    for key in ("_timestamp", "timestamp", "time", "t"):
        value = row.get(key)
        if number(value) and value > 0:
            return float(value)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    return parsed.timestamp()
            except ValueError:
                pass
    return None


def version(row):
    for key in ("policy/version", "_step", "step", "update"):
        value = integer(row.get(key))
        if value is not None:
            return value
    return None


def metric_row(row):
    result = dict(row)
    duration = row.get("seconds", row.get("timing/round_seconds", row.get("train/step_time")))
    if number(duration) and duration >= 0:
        result["train/step_time"] = duration
    return result


def has_metrics(row):
    return any(number(value) for key, value in row.items()
               if key in CORE_KEYS or key.startswith(MORE_PREFIXES) and key != "policy/version")


def read_json(path, warnings):
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            return value
        warnings.append(f"{path.name} is not a JSON object.")
    except (OSError, ValueError) as exc:
        warnings.append(f"Could not read {path.name}: {type(exc).__name__}.")
    return {}


def read_jsonl(path, warnings):
    if not path.is_file():
        return []
    rows, skipped = [], 0
    try:
        with path.open(errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                    else:
                        skipped += 1
                except ValueError:
                    skipped += 1
    except OSError as exc:
        warnings.append(f"Could not read {path.name}: {type(exc).__name__}.")
    if skipped:
        warnings.append(f"Ignored {skipped} incomplete or invalid line(s) in {path.name}.")
    if any(clean(row) != row for row in rows):
        warnings.append(f"Nonfinite values in {path.name} are displayed as missing.")
    return [clean(row) for row in rows]


def fold_metrics(rows):
    """Preserve file order: equal update merges; a lower update rewinds."""
    result = {}
    previous = None
    for raw in rows:
        row = metric_row(raw)
        update = version(row)
        if update is None or not has_metrics(row):
            continue
        if previous is not None and update < previous:
            result = {step: item for step, item in result.items() if step < update}
        result[update] = {**result.get(update, {}), **row}
        previous = update
    return result


def fold_events(rows):
    """Keep phase/profile/checkpoint evidence on the active append-log branch."""
    history, profiles, checkpoints = {}, {}, []
    active_update, phase, rollout = None, None, None
    observed, accepted = None, []

    def rewind(update):
        nonlocal history, profiles, checkpoints, phase, rollout
        history = {step: row for step, row in history.items() if step < update}
        profiles = {step: row for step, row in profiles.items() if step < update}
        checkpoints = [row for row in checkpoints if row["step"] < update]
        phase = rollout = None

    for row in rows:
        # The file name also declares rank zero; explicitly different ranks are
        # rejected so rank-local profiling can never look globally reduced.
        if row.get("rank", 0) != 0:
            continue
        row = {**row, "rank": 0}
        accepted.append(row)
        observed = timestamp(row) or observed
        event, update = row.get("event"), version(row)
        if event == "resumed":
            cursor = integer(row.get("next_update"))
            if cursor is not None:
                rewind(cursor + 1)
                active_update = cursor
            continue
        if update is not None and event in {"phase", "rollout_progress", "profile", "update"}:
            if active_update is not None and update < active_update:
                rewind(update)
            elif event == "phase" and row.get("phase") == "data_audio_begin" and update in history:
                rewind(update)
            active_update = update
        if event == "update" and update is not None and has_metrics(metric_row(row)):
            history[update] = {**history.get(update, {}), **metric_row(row)}
            phase = rollout = None
        elif event == "profile" and update is not None:
            profiles[update] = {**profiles.get(update, {}), **row}
        elif event == "phase":
            phase = {"name": row.get("phase"), "time": timestamp(row), "rank": 0,
                     **{key: row[key] for key in ("update", "role", "microbatch", "rows", "packed_positions") if key in row}}
            if not str(row.get("phase", "")).startswith("actor_"):
                rollout = None
        elif event == "rollout_progress" and update is not None:
            if number(row.get("current_frame")) and number(row.get("total_frames")) and row["total_frames"] > 0:
                rollout = {**row, "kind": "submitted_frames"}
        elif event == "checkpoint" and isinstance(row.get("path"), dict):
            path = row["path"]
            step = integer(path.get("step"))
            if step is not None:
                checkpoints.append({"step": step, "time": timestamp(row),
                                    **{key: path.get(key) for key in ("path", "resume_path", "export_path")},
                                    "evidence": "events-rank0.jsonl"})
                phase = rollout = None
    return {"history": history, "profiles": profiles, "checkpoints": checkpoints,
            "active_update": active_update, "phase": phase, "rollout_progress": rollout,
            "observed_at": observed, "latest_event": accepted[-1] if accepted else None,
            "recent_events": accepted[-20:]}


def metadata(root, overrides, warnings):
    preparation = read_json(root / "preparation.json", warnings) if root else {}
    config = read_json(root / "config.json", warnings) if root else {}
    plan = read_json(root / "plan.json", warnings) if root else {}
    prepared_plan = preparation.get("plan")
    prepared_identity = preparation.get("identity")
    if not config and isinstance(prepared_identity, dict):
        config = prepared_identity
    if not plan and isinstance(prepared_plan, dict):
        # A preparation made with a different launch world/config is not the
        # current run's plan. Actual config.json wins over preparation.json.
        if prepared_identity and config != prepared_identity:
            warnings.append("Preparation identity differs from config.json; its plan was ignored.")
        else:
            plan = prepared_plan
    method = config.get("method", {})
    method = method if isinstance(method, dict) else {}
    updates = plan.get("updates")
    updates = updates if isinstance(updates, list) else None
    max_steps = len(updates) if updates is not None else None
    epoch_rows = integer(plan.get("rows_per_epoch"), minimum=1)
    effective_batch = integer(plan.get("effective_batch"), minimum=1)
    world = integer(plan.get("world", config.get("world")), minimum=1)
    rows_per_rank = integer(plan.get("rows_per_rank", method.get("effective_batch_per_rank")), minimum=1)
    if effective_batch is None and world is not None and rows_per_rank is not None:
        effective_batch = world * rows_per_rank
    if effective_batch is None and updates and all(isinstance(row, list) for row in updates):
        lengths = {len(row) for row in updates}
        if len(lengths) == 1:
            effective_batch = integer(next(iter(lengths)), minimum=1)
    seconds = method.get("checkpoint_interval_seconds")
    every_steps = integer(method.get("checkpoint_interval_steps"), minimum=1)
    export_steps = list(range(every_steps, max_steps + 1, every_steps)) if every_steps and max_steps else []
    if max_steps and (not export_steps or export_steps[-1] != max_steps):
        export_steps.append(max_steps)
    result = {"max_steps": max_steps, "rows_per_step": effective_batch, "epoch_rows": epoch_rows,
              "save_every_min": seconds / 60 if number(seconds) and seconds > 0 else None,
              "save_every_steps": every_steps, "save_wall": bool(number(seconds) and seconds > 0),
              "export_steps": export_steps,
              "learning_rate_peak": method.get('learning_rate'),
              "learning_rate_schedule": config.get('optimizer', {}).get('schedule'),
              "world": world, "epochs": integer(plan.get("epochs", method.get("epochs")), minimum=1)}
    for name in ("max_steps", "rows_per_step", "epoch_rows", "save_every_min"):
        if overrides.get(name) is not None:
            value = overrides[name]
            valid = number(value) and value > 0 and (name == "save_every_min" or value == int(value))
            if not valid:
                raise ValueError(f"{name} override must be a positive {'number' if name == 'save_every_min' else 'integer'}")
            result[name] = value if name == "save_every_min" else int(value)
    return result, updates


def build_doc(run_dir: Path | None, overrides: dict | None = None) -> dict:
    overrides = dict(overrides or {})
    root = Path(run_dir).expanduser().resolve() if run_dir is not None else None
    warnings = []
    if root and not root.is_dir():
        warnings.append("Run directory is not available yet.")
    meta, plan_updates = metadata(root, overrides, warnings)
    metric_rows = read_jsonl(root / "metrics.jsonl", warnings) if root else []
    events = read_jsonl(root / "events-rank0.jsonl", warnings) if root else []
    metrics, telemetry = fold_metrics(metric_rows), fold_events(events)
    event_history, active_update = telemetry["history"], telemetry["active_update"]
    history = dict(metrics)
    if event_history or active_update is not None:
        # Read metrics first and events second: the trainer writes each update
        # event before its metrics line. Rank-zero events therefore establish
        # the newest branch, including a restart before its first new update.
        ceiling = max(event_history, default=-1)
        if active_update is not None and active_update not in event_history:
            ceiling = active_update - 1
        history = {step: row for step, row in metrics.items() if step <= ceiling}
        for step, event_row in event_history.items():
            other = metrics.get(step, {})
            common = [key for key in other if key in event_row and (key in CORE_KEYS or key.startswith(MORE_PREFIXES))]
            conflict = any(other[key] != event_row[key] for key in common)
            if conflict:
                warnings.append(f"Update {step} differs between logs; timestamped rank-zero event was used.")
                history[step] = event_row
            else:
                history[step] = {**other, **event_row}
        if any(step > ceiling for step in metrics):
            warnings.append("Metrics beyond the current rank-zero event history were ignored while the logs converge.")
    history = dict(sorted(history.items()))
    keys = [key for key in CORE_KEYS if any(number(row.get(key)) for row in history.values())]
    keys.extend(sorted({key for row in history.values() for key, value in row.items()
                        if key.startswith(MORE_PREFIXES) and key != "policy/version" and number(value)} - set(keys)))
    rows_per_step = meta["rows_per_step"]

    def row_count(step):
        return step * rows_per_step if rows_per_step is not None else None

    train = [[step, row_count(step)] + [row.get(key) if number(row.get(key)) else None for key in keys]
             for step, row in history.items()]
    step = max(history, default=None)
    meta['next_export_step'] = next((value for value in meta['export_steps'] if value > (step or 0)), None)
    times = [timestamp(row) for row in history.values()]
    durations = [row.get("train/step_time") for row in history.values()
                 if number(row.get("train/step_time")) and row["train/step_time"] > 0]
    recent_duration = statistics.median(durations[-10:]) if durations else None
    checkpoint = telemetry["checkpoints"][-1] if telemetry["checkpoints"] else None
    profile_rows = telemetry["profiles"]
    profile_keys = sorted({key for row in profile_rows.values() for key, value in row.items()
                           if key not in {"time", "rank", "update", "step", "_step", "policy/version"} and number(value)})
    profiles = [[update, row_count(update)] + [row.get(key) if number(row.get(key)) else None for key in profile_keys]
                for update, row in sorted(profile_rows.items())]
    result = read_json(root / "RESULT.json", warnings) if root else {}
    state = "observed" if history or telemetry["latest_event"] else "pending"
    terminal = result.get("status") in {"completed", "bounded_or_stopped"}
    result_update = integer(result.get("next_update"))
    if terminal:
        result_mtime = (root / "RESULT.json").stat().st_mtime
        stale = (telemetry["observed_at"] is not None and telemetry["observed_at"] > result_mtime) or (step is not None and result_update != step)
        if stale:
            warnings.append("RESULT.json predates the current telemetry; its terminal status was ignored.")
        else:
            state = result["status"]
            if step is None:
                step = result_update
    supervisor = read_json(root / 'supervisor' / 'finished.json', warnings) if root else {}
    exit_code = supervisor.get('exit_code')
    if isinstance(exit_code,int) and not isinstance(exit_code,bool) and exit_code != 0:
        state='failed'
        warnings.append(f'Training supervisor exited with code {exit_code}; this attempt is not running.')
    now = datetime.now(timezone.utc)
    doc = {
        "run": overrides.get("run") or (root.name if root else "pose_opd_credit"),
        "run_dir": str(root) if root else None, "src": str(root / "metrics.jsonl") if root else None,
        "jobs": [], "job_runs": {}, "started": bool(history or telemetry["latest_event"]),
        "state": state, "done": state == "completed", "step": step, **meta,
        "supervisor_exit_code": exit_code, "supervisor_finished_at": supervisor.get('time'),
        "learning_rate_current": history.get(step, {}).get('learning_rate') if step is not None else None,
        "train_cols": ["step", "rows"] + keys, "train": train, "cols": ["step", "rows"], "pts": [],
        "update_times": times, "t_last": times[-1] if times else None,
        "t_step1": times[0] if times else None, "mean50_s": recent_duration, "mean_s": recent_duration,
        "duration_estimator": "median of last 10 recorded update durations",
        "last_save": [checkpoint["time"], checkpoint["step"]] if checkpoint else None,
        "checkpoint": checkpoint, "checkpointing_enabled": result.get("checkpointing_enabled") if terminal and state != "observed" else None,
        "profile_cols": ["step", "rows"] + profile_keys, "profile": profiles, "profile_rank": 0,
        "profile_scope": "rank 0 only", "phase": telemetry["phase"],
        "rollout_progress": telemetry["rollout_progress"], "latest_event": telemetry["latest_event"],
        "recent_events": telemetry["recent_events"], "observed_at": telemetry["observed_at"],
        "updated": now.isoformat(), "generated_at": now.timestamp(), "warnings": list(dict.fromkeys(warnings)),
        "rows_axis": "planned slots (includes zero-weight tail padding)",
    }
    # Report actual non-padding coverage separately, never rename slots as rows.
    if step is not None and plan_updates is not None and all(isinstance(row, list) for row in plan_updates):
        doc["processed_rows"] = sum(item is not None for batch in plan_updates[:step] for item in batch)
    else:
        doc["processed_rows"] = None
    return clean(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    doc = build_doc(args.run_dir)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(doc, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()

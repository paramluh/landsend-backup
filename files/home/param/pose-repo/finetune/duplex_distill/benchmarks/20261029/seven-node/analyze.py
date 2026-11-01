"""Rebuild timing-summary.json from the accompanying, unmodified profile records."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = (3, 4, 5, 6)


def stats(values):
    values = list(values)
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def analyze(directory):
    raw = json.loads((directory / "profile-all-ranks.json").read_text())
    cfg = json.loads((directory / "config.json").read_text())
    records = [record for rank in raw["records"] for record in rank]
    steady = [record for record in records if record["step"] in STEPS]
    world, mbs = cfg["world"], cfg["micro_batch_size"]
    assert world == 56 and len(records) == 336 and len(steady) == 224
    assert all(sum(r["step"] == step for r in records) == world for step in range(1, 7))
    assert all(r["row_count"] == mbs for r in records)
    assert all(not row["dummy"] for r in records for row in r["per_row_counts"])
    per_step = []
    for step in range(1, 7):
        rows = [r for r in records if r["step"] == step]
        per_step.append({"step": step,
            "max_rank_wall_seconds": max(r["wall_seconds"] for r in rows),
            "mean_rank_wall_seconds": statistics.mean(r["wall_seconds"] for r in rows),
            "min_rank_wall_seconds": min(r["wall_seconds"] for r in rows),
            "real_rows": sum(r["real_row_count"] for r in rows),
            "conversation_frames": sum(r["text_positions"] + r["real_row_count"] for r in rows),
            "timeline_frames": sum(r["timeline_frames"] for r in rows),
            "speech_depth_frames": sum(r["speech_depth_frames"] for r in rows),
            "pose_depth_frames": sum(r["pose_depth_frames"] for r in rows)})
    selected = [r for r in per_step if r["step"] in STEPS]
    wall = sum(r["max_rank_wall_seconds"] for r in selected)
    stages = {}
    for key in sorted({key for record in steady for key in record["stages"]}):
        stages[key] = {field: stats(r["stages"].get(key, {}).get(field, 0) for r in steady)
                       for field in ("gpu_seconds", "cpu_dispatch_seconds", "calls")}
    totals = {key: sum(r[key] for r in selected) for key in (
        "real_rows", "conversation_frames", "timeline_frames", "speech_depth_frames", "pose_depth_frames")}
    parents = [key for key in stages if "/" not in key]
    row_records = [row for r in steady for row in r["per_row_counts"]]
    valid_pose = [r for r in row_records if r["pose_depth_frames"]]
    assert all(r["pose_depth_frames"] == r["speech_depth_frames"] for r in valid_pose)
    mean_wall = wall / len(STEPS)
    return {
        "micro_batch_size": mbs, "world": world, "global_batch_size": cfg["global_batch_size"],
        "schedule_steps": cfg["steps"], "warmup_steps": cfg["warmup"],
        "text_head": cfg["text_head"], "total_rank_steps": len(records),
        "steady_rank_steps": len(steady), "steady_steps": STEPS,
        "per_step": per_step,
        "mean_step_max_rank_wall_seconds": mean_wall,
        "rank_wall_seconds": stats(r["wall_seconds"] for r in steady),
        "steady_totals": totals,
        "global_throughput_per_second": {key: value / wall for key, value in totals.items()},
        "peak_allocated_GB_all_six_steps": max(r["max_memory_GB"] for r in records),
        "peak_allocated_GB_steady_steps": max(r["max_memory_GB"] for r in steady),
        "projected_two_epoch_seconds": mean_wall * cfg["steps"],
        "projected_two_epoch_hours": mean_wall * cfg["steps"] / 3600,
        "steady_valid_pose_rows": len(valid_pose),
        "steady_missing_pose_rows": len(row_records) - len(valid_pose),
        "all_six_steps_real_rows": sum(r["real_row_count"] for r in records),
        "all_six_steps_valid_pose_rows": sum(bool(row["pose_depth_frames"]) for r in records for row in r["per_row_counts"]),
        "sum_parent_mean_gpu_seconds": sum(stages[key]["gpu_seconds"]["mean"] for key in parents),
        "mean_rank_wall_minus_parent_sum_seconds": statistics.mean(r["wall_seconds"] for r in steady)
            - sum(stages[key]["gpu_seconds"]["mean"] for key in parents),
        "stages": stages,
        "source_file_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()},
    }


if __name__ == "__main__":
    output = {"method": {
        "primary_window": "steps 3 through 6, 56 ranks: 224 rank-steps per run",
        "wall": "mean across four steps of each step's maximum rank wall_seconds",
        "throughput": "global valid row/frame totals divided by sum of per-step maximum rank wall time",
        "conversation_frames": "text_positions + real_row_count; source selects n-1 text positions per non-dummy row",
        "stages": "CUDA event intervals, averaged across 224 rank-steps; include CPU dispatch gaps and communication/arrival waits; nested intervals overlap parents",
        "projection": "configured optimizer update count multiplied by steady mean maximum-rank wall; excludes loading, save cost, and changing dataset composition",
        "memory": "maximum reported CUDA allocated high-water mark; not reserved/device-total memory",
    }, "runs": {}}
    for name in ("mbs1", "mbs2"):
        if (ROOT / name / "profile-all-ranks.json").exists():
            output["runs"][name] = analyze(ROOT / name)
    if len(output["runs"]) == 2:
        a, b = output["runs"]["mbs1"], output["runs"]["mbs2"]
        output["comparison"] = {
            "mbs2_vs_mbs1_update_time_ratio": b["mean_step_max_rank_wall_seconds"] / a["mean_step_max_rank_wall_seconds"],
            "mbs2_vs_mbs1_row_throughput_ratio": b["global_throughput_per_second"]["real_rows"] / a["global_throughput_per_second"]["real_rows"],
            "mbs2_vs_mbs1_conversation_frame_throughput_ratio": b["global_throughput_per_second"]["conversation_frames"] / a["global_throughput_per_second"]["conversation_frames"],
            "mbs2_projected_time_reduction_percent": 100 * (1 - b["projected_two_epoch_seconds"] / a["projected_two_epoch_seconds"]),
            "mbs2_minus_mbs1_peak_allocated_GB": b["peak_allocated_GB_all_six_steps"] - a["peak_allocated_GB_all_six_steps"],
        }
    (ROOT / "timing-summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {key: data[key] for key in ("mean_step_max_rank_wall_seconds", "global_throughput_per_second", "peak_allocated_GB_all_six_steps", "projected_two_epoch_hours")}
                      for name, data in output["runs"].items()}, indent=2))
    if "comparison" in output:
        print(json.dumps(output["comparison"], indent=2))

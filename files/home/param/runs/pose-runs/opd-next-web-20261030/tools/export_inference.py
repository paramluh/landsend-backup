"""CPU-only conversion of a committed OPD checkpoint into native audio inference shards.

The input is a flat model DCP directory containing every rank's model files and
rank zero's .metadata, without optimizer files. Legacy lexical OPD exports reuse
the pinned base's frozen speech modules; --full-model requires all trained audio
weights and uses the base only for architecture configuration. Trained tensors use
the actor publication cast: BF16 weights and FP32 vgain.theta. No GPU is required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.distributed.checkpoint as dcp


TRAINABLE = ("inp.", "layers.", "head.", "vadapter.", "vgain.")
FROZEN = ("tower_s.", "dec_s.")
IGNORED_BUFFERS = {"vadapter.target_rms"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message):
    print(message, flush=True)


def _load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def _read_manifest(manifest_path, model_dcp):
    """Accept the atomic weights-only epoch exports made by our SFT trainer."""
    manifest = json.loads(Path(manifest_path).read_text())
    manifest = manifest.get("manifest", manifest)
    if "source_checkpoint" not in manifest:
        return manifest
    if (manifest.get("format") != "opd-dcp-v1" or
            manifest.get("weights_dtype") != "float32" or
            manifest.get("optimizer_saved") is not False or
            manifest.get("epoch") not in (1, 2) or
            manifest.get("step") != 27 * manifest["epoch"]):
        raise ValueError("Expected this run's completed FP32 weights-only epoch checkpoint")
    source = manifest["source_checkpoint"]
    parent_path = Path(source["path"]).parent / "MANIFEST.json"
    parent = json.loads(parent_path.read_text())
    if parent["checkpoint_id"] != source["checkpoint_id"]:
        raise ValueError("Finetune initialization identity does not match its native source")
    metadata = dcp.FileSystemReader(model_dcp).read_metadata()
    files = sorted({".metadata"} | {item.relative_path for item in metadata.storage_data.values()})
    normalized = dict(manifest, student_version=parent["student_version"],
                      base_checkpoint=parent["base_checkpoint"], _finetune=manifest,
                      files=[{"path": "model/" + name, "bytes": (Path(model_dcp) / name).stat().st_size}
                             for name in files])
    return normalized


def _verify_source(model_dcp: Path, manifest: dict):
    if manifest.get("format") == "opd-dcp-v1":
        snapshot = model_dcp.parent
        manifest_digest = sha256(snapshot / "MANIFEST.json")
        if manifest.get("_finetune"):
            if (snapshot / "COMPLETE").read_text() != "model weights only\n":
                raise ValueError("Finetune epoch export has no completed weights-only commit marker")
            complete = {"manifest_sha256": manifest_digest}
        elif (snapshot / "DOWNLOAD_VERIFIED.json").is_file():
            downloaded = json.loads((snapshot / "DOWNLOAD_VERIFIED.json").read_text())
            if (downloaded.get("format") != "opd-model-only-hf-download-v1" or
                    downloaded.get("checkpoint_id") != manifest["checkpoint_id"] or
                    downloaded.get("manifest_sha256") != manifest_digest or
                    not downloaded.get("revision") or not downloaded.get("model_only")):
                raise ValueError("Pinned HF model download identity/manifest verification failed")
            complete = json.loads((snapshot / "COMPLETE").read_text())
        else:
            if not (snapshot / "SNAPSHOT_COMPLETE").is_file():
                raise ValueError("Rolling checkpoint must be a completed isolated snapshot")
            complete = json.loads((snapshot / "SOURCE_COMPLETE").read_text())
        if complete["manifest_sha256"] != manifest_digest:
            raise ValueError("Source COMPLETE does not match original manifest hash")
    expected = {}
    for item in manifest["files"]:
        if not item["path"].startswith("model/"):
            continue
        relative = Path(item["path"]).relative_to("model")
        if relative.is_absolute() or ".." in relative.parts or str(relative) in expected:
            raise ValueError("Unsafe or duplicate model shard path")
        path = model_dcp / relative
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Model shard differs from committed manifest: {relative}")
        digest = sha256(path)
        if item.get("sha256") and digest != item["sha256"]:
            raise ValueError(f"Model shard differs from committed manifest: {relative}")
        verified = dict(item, actual_sha256=digest)
        expected[str(relative)] = verified
        log(f"Verified {relative}: {path.stat().st_size} bytes, sha256={digest}")
    if ".metadata" not in expected:
        raise ValueError("Manifest lacks model DCP metadata")
    metadata = dcp.FileSystemReader(model_dcp).read_metadata()
    for location in metadata.storage_data.values():
        item = expected.get(location.relative_path)
        if item is None or location.offset + location.length > item["bytes"]:
            raise ValueError("DCP references an unverified or truncated model shard")
    verification = model_dcp.parent / "INFERENCE_SOURCE_VERIFIED.json"
    verification.write_text(json.dumps({"format": "opd-inference-source-verification-v1", "manifest_format": manifest["format"], "checkpoint_id": manifest["checkpoint_id"], "files": list(expected.values())}, indent=2) + "\n")
    return metadata


def _base_layout(base: Path, world: int):
    shapes, replicas = {}, {}
    for rank in range(world):
        state = _load(base / f"rank{rank}.pt")
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
                raise ValueError(f"Expected a dim-zero native tensor shard: {name}")
            if not name.startswith(TRAINABLE + FROZEN):
                raise ValueError(f"Unexpected actor module in pinned base: {name}")
            shape = shapes.setdefault(name, [0, *tensor.shape[1:]])
            if tuple(shape[1:]) != tuple(tensor.shape[1:]):
                raise ValueError(f"Inconsistent native shard shape: {name}")
            shape[0] += tensor.shape[0]
        del state
    state = _load(base / "rep.pt")
    for name, tensor in state.items():
        if name in shapes or not name.startswith(TRAINABLE + FROZEN):
            raise ValueError(f"Unexpected or duplicate replicated actor tensor: {name}")
        replicas[name] = tuple(tensor.shape)
    shapes.update(replicas)
    return {k: tuple(v) for k, v in shapes.items()}, set(replicas)


def export_inference(*, model_dcp: Path, manifest_path: Path, base: Path,
                     output: Path, student_version: str) -> dict:
    model_dcp, manifest_path, base, output = map(Path, (model_dcp, manifest_path, base, output))
    manifest = _read_manifest(manifest_path, model_dcp)
    if manifest.get("format") not in ("opd-hf-sharded-v1", "opd-dcp-v1") or str(manifest.get("student_version")) != str(student_version):
        raise ValueError("Expected a committed OPD manifest with the requested student version")
    if not manifest.get("checkpoint_id") or not manifest.get("base_checkpoint"):
        raise ValueError("Checkpoint identity and pinned actor base are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty export: {output}")
    base_meta = json.loads((base / "meta.json").read_text())
    if base_meta.get("format") != "ppm-local-bf16-shards-v1" or not (base / "COMPLETE").is_file():
        raise ValueError("Expected a complete native actor base")
    world = int(base_meta["world"])
    if world < 1:
        raise ValueError("Native base needs a positive shard count")
    log("Verifying committed OPD model file hashes (CPU only)")
    metadata = _verify_source(model_dcp, manifest)
    shapes, replicas = _base_layout(base, world)
    expected = {name: shape for name, shape in shapes.items() if name.startswith(TRAINABLE)}
    if any(not any(name.startswith(group) for name in expected) for group in TRAINABLE):
        raise ValueError("Native base is missing a required learner module")
    if any(not any(name.startswith(group) for name in shapes) for group in FROZEN):
        raise ValueError("Native base is missing a required frozen audio module")
    stored = {}
    for key, item in metadata.state_dict_metadata.items():
        if not key.startswith("model.") or not hasattr(item, "size"):
            raise ValueError(f"Unexpected DCP state key: {key}")
        stored[key[len("model."):]] = item
    if set(stored) - set(expected) - IGNORED_BUFFERS or set(expected) - set(stored):
        raise ValueError("OPD/base tensor mapping mismatch: " + str({
            "missing": sorted(set(expected) - set(stored)),
            "unexpected": sorted(set(stored) - set(expected) - IGNORED_BUFFERS)}))
    for name, shape in expected.items():
        if tuple(stored[name].size) != shape:
            raise ValueError(f"OPD/base shape mismatch: {name}")
    if expected["vgain.theta"] != (1,):
        raise ValueError("Expected the single-element adapter log-gain")
    weights = {name: torch.empty(shape, dtype=torch.float32 if name == "vgain.theta" else torch.bfloat16)
               for name, shape in expected.items()}
    buffers = {name: torch.empty(tuple(stored[name].size), dtype=torch.float32)
               for name in IGNORED_BUFFERS & set(stored)}
    log(f"Loading {len(weights)} learner tensors; casting exactly as actor publication")
    dcp.load({"model": {**weights, **buffers}}, checkpoint_id=model_dcp, no_dist=True)
    for name, value in {**weights, **buffers}.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite OPD tensor: {name}")
    if "vadapter.target_rms" in buffers and not torch.equal(buffers["vadapter.target_rms"], torch.ones_like(buffers["vadapter.target_rms"])):
        raise ValueError("Unexpected adapter normalization buffer; native export assumes norm='none'")
    theta = float(weights["vgain.theta"].item())
    if not math.isfinite(math.exp(theta)):
        raise ValueError("Nonfinite adapter gain")
    output.mkdir(parents=True, exist_ok=True)
    offsets = {name: 0 for name in expected if name not in replicas}
    files = []
    for filename in [*(f"rank{rank}.pt" for rank in range(world)), "rep.pt"]:
        source = _load(base / filename)
        converted = {}
        for name, value in source.items():
            if name in weights:
                if name in replicas:
                    selected = weights[name]
                else:
                    start = offsets[name]
                    selected = weights[name][start:start + value.shape[0]]
                    offsets[name] += value.shape[0]
                if selected.shape != value.shape:
                    raise ValueError(f"Invalid slice while exporting {name}")
                converted[name] = selected.clone().contiguous()
            else:
                converted[name] = value.clone().contiguous()
        destination = output / filename
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(converted, temporary)
        temporary.replace(destination)
        # Verify actual serialized bytes decode to exactly the intended tensors.
        checked = _load(destination)
        if checked.keys() != converted.keys() or any(
                checked[name].dtype != value.dtype or not torch.equal(checked[name], value)
                for name, value in converted.items()):
            raise ValueError(f"Export tensor round-trip failed: {filename}")
        files.append(dict(path=filename, bytes=destination.stat().st_size, sha256=sha256(destination)))
        del checked, converted, source
        log(f"Wrote and verified {filename}")
    if any(offsets[name] != expected[name][0] for name in offsets):
        raise ValueError("Incomplete learner tensor coverage after native shard export")
    # Old base run metadata must not cause live_ppm to restore an old gain.
    cfg = copy.deepcopy(base_meta["cfg"])
    cfg.pop("log_root", None)
    cfg.pop("run_tag", None)
    provenance = dict(
        format="opd-native-inference-export-v1", checkpoint_id=manifest["checkpoint_id"],
        student_version=str(student_version), base_checkpoint=manifest["base_checkpoint"],
        base_export_step=base_meta["step"], base_path=str(base.resolve()),
        source_manifest_sha256=sha256(manifest_path), base_meta_sha256=sha256(base / "meta.json"),
        model_dcp=str(model_dcp.resolve()), learner_tensors=len(weights),
        frozen_prefixes=list(FROZEN), tensor_precision="BF16 except FP32 vgain.theta",
        vgain_theta_fp32=theta, adapter_gain=math.exp(theta), files=files,
        config_fingerprint=manifest.get("config_fingerprint"), counters=manifest.get("counters"))
    meta = dict(format="ppm-local-bf16-shards-v1", world=world, step=int(student_version), cfg=cfg,
                opd_student_version=str(student_version), base_export_step=base_meta["step"],
                checkpoint_id=manifest["checkpoint_id"], source="OPD learner plus pinned frozen actor base")
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "COMPLETE").write_text(json.dumps(dict(
        student_version=str(student_version), provenance_sha256=sha256(output / "PROVENANCE.json"))) + "\n")
    log(f"COMPLETE: {output}; OPD update {student_version}; trained adapter gain {math.exp(theta):.12g}")
    return provenance


def _actor_shapes(base_meta):
    """Construct only meta tensors to verify the complete speech inference model."""
    model_dir = Path("/home/param/pose-repo/finetune/duplex_distill/production")
    sys.path.insert(0, str(model_dir))
    import model_ppm as model
    model.QWEN32 = "/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B"
    model.QWEN4B_CFG = "/home/param/pose-repo/infer/engine/ppm/configs/Qwen3-4B"
    cfg = base_meta["cfg"]
    dims = ({k: int(cfg[k]) for k in ("layers", "tower_layers", "dec_layers", "proj_r", "tower_vocab")}
            if "layers" in cfg else dict(layers=64, tower_layers=36, dec_layers=28, proj_r=2048,
                                        tower_vocab=model.TOWER_VOCAB))
    with torch.device("meta"):
        actor = model.PPM(model.cfg32(dims["layers"]), model.cfg4b(dims["tower_layers"]),
                          model.cfg06(dims["dec_layers"]), dims["proj_r"], dims["tower_vocab"],
                          vadapter=dict(enc_hidden=1280, hidden=4096, stack=4, norm="none"), nopose=True)
    return {name: tuple(value.shape) for name, value in actor.named_parameters()}


def export_full_inference(*, model_dcp: Path, manifest_path: Path, base: Path,
                          output: Path, student_version: str, shards: int = 16) -> dict:
    """Export a speech-preservation learner, including its trained audio modules.

    The original export supplies architecture configuration only. Every inference
    parameter must be present in the new checkpoint; no original audio weights
    are substituted. Native shards limit conversion/serialization peak RAM.
    """
    model_dcp, manifest_path, base, output = map(Path, (model_dcp, manifest_path, base, output))
    manifest = _read_manifest(manifest_path, model_dcp)
    if (manifest.get("format") not in ("opd-hf-sharded-v1", "opd-dcp-v1") or
            str(manifest.get("student_version")) != str(student_version)):
        raise ValueError("Expected a committed OPD manifest with the requested student version")
    if shards < 1 or not manifest.get("checkpoint_id") or not manifest.get("base_checkpoint"):
        raise ValueError("Checkpoint identity, pinned base and positive shard count are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty export: {output}")
    base_meta = json.loads((base / "meta.json").read_text())
    if base_meta.get("format") != "ppm-local-bf16-shards-v1" or not (base / "COMPLETE").is_file():
        raise ValueError("Expected complete native base configuration")
    log("Verifying all committed model shards, including trained speech tower and depth decoder")
    metadata = _verify_source(model_dcp, manifest)
    shapes = _actor_shapes(base_meta)
    if any(not any(name.startswith(group) for name in shapes) for group in TRAINABLE + FROZEN):
        raise ValueError("Inference architecture is missing a required speech module")
    stored = {}
    for key, item in metadata.state_dict_metadata.items():
        if not key.startswith("model.") or not hasattr(item, "size"):
            raise ValueError(f"Unexpected DCP state key: {key}")
        stored[key[len("model."):]] = item
    if set(stored) - set(shapes) - IGNORED_BUFFERS or set(shapes) - set(stored):
        raise ValueError("Full OPD/actor mapping mismatch: " + str({
            "missing": sorted(set(shapes)-set(stored)),
            "unexpected": sorted(set(stored)-set(shapes)-IGNORED_BUFFERS)}))
    for name, shape in shapes.items():
        if tuple(stored[name].size) != shape:
            raise ValueError(f"OPD/actor shape mismatch: {name}")
    if shapes["vgain.theta"] != (1,):
        raise ValueError("Expected the single-element adapter log-gain")
    weights = {name: torch.empty(shape, dtype=torch.float32 if name == "vgain.theta" else torch.bfloat16)
               for name, shape in shapes.items()}
    buffers = {name: torch.empty(tuple(stored[name].size), dtype=torch.float32)
               for name in IGNORED_BUFFERS & set(stored)}
    log(f"Loading all {len(weights)} trained inference tensors; BF16 with FP32 adapter gain")
    dcp.load({"model": {**weights, **buffers}}, checkpoint_id=model_dcp, no_dist=True)
    for name, value in {**weights, **buffers}.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite OPD tensor: {name}")
    if "vadapter.target_rms" in buffers and not torch.equal(buffers["vadapter.target_rms"], torch.ones_like(buffers["vadapter.target_rms"])):
        raise ValueError("Unexpected adapter normalization buffer")
    theta = float(weights["vgain.theta"].item())
    if not math.isfinite(math.exp(theta)):
        raise ValueError("Nonfinite adapter gain")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for rank in range(shards + 1):
        filename = f"rank{rank}.pt" if rank < shards else "rep.pt"
        converted = {}
        for name, value in weights.items():
            if rank == shards:
                if value.ndim == 0: converted[name] = value.clone()
            elif value.ndim:
                width = math.ceil(value.shape[0] / shards)
                converted[name] = value[min(rank*width, value.shape[0]):min((rank+1)*width, value.shape[0])].clone().contiguous()
        target = output / filename
        temporary = target.with_suffix(".pt.tmp")
        torch.save(converted, temporary)
        temporary.replace(target)
        checked = _load(target)
        if checked.keys() != converted.keys() or any(checked[name].dtype != value.dtype or
                not torch.equal(checked[name], value) for name, value in converted.items()):
            raise ValueError(f"Export tensor round-trip failed: {filename}")
        files.append(dict(path=filename, bytes=target.stat().st_size, sha256=sha256(target)))
        del checked, converted
        log(f"Wrote and verified {filename}")
    cfg = copy.deepcopy(base_meta["cfg"])
    cfg.pop("log_root", None); cfg.pop("run_tag", None)
    provenance = dict(format="opd-native-inference-export-v1", checkpoint_id=manifest["checkpoint_id"],
        student_version=str(student_version), base_checkpoint=manifest["base_checkpoint"],
        base_export_step=base_meta["step"], base_path=str(base.resolve()),
        source_manifest_sha256=sha256(manifest_path), base_meta_sha256=sha256(base / "meta.json"),
        model_dcp=str(model_dcp.resolve()), learner_tensors=len(weights), frozen_prefixes=[],
        trained_prefixes=list(TRAINABLE + FROZEN), tensor_precision="BF16 except FP32 vgain.theta",
        vgain_theta_fp32=theta, adapter_gain=math.exp(theta), files=files,
        config_fingerprint=manifest.get("config_fingerprint"), counters=manifest.get("counters"))
    meta = dict(format="ppm-local-bf16-shards-v1", world=shards, step=int(student_version), cfg=cfg,
        opd_student_version=str(student_version), base_export_step=base_meta["step"],
        checkpoint_id=manifest["checkpoint_id"], vgain_theta_fp32=theta, source="Complete trained OPD speech-preservation learner", source_manifest_format=manifest["format"], credit_lookback_words=1)
    if manifest.get("_finetune"):
        ft = manifest["_finetune"]
        meta.update(step=ft["step"], finetune_epoch=ft["epoch"],
                    source="Rosie assistant-only finishing SFT from OPD v110; complete epoch weights")
        provenance.update(finetune_epoch=ft["epoch"], finetune_step=ft["step"],
                          finetune_manifest=ft)
    (output / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (output / "COMPLETE").write_text(json.dumps(dict(student_version=str(student_version),
        provenance_sha256=sha256(output / "PROVENANCE.json")))+"\n")
    log(f"COMPLETE: {output}; all trained speech modules included; gain {math.exp(theta):.12g}")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model-dcp", "manifest", "base", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--student-version", required=True)
    parser.add_argument("--full-model", action="store_true", help="Require trained audio tower/decoder; use base configuration only")
    parser.add_argument("--shards", type=int, default=16, help="Native output shards for --full-model")
    args = parser.parse_args()
    torch.set_num_threads(8)
    function = export_full_inference if args.full_model else export_inference
    function(model_dcp=args.model_dcp, manifest_path=args.manifest, base=args.base,
             output=args.output, student_version=args.student_version,
             **({"shards": args.shards} if args.full_model else {}))


if __name__ == "__main__":
    main()

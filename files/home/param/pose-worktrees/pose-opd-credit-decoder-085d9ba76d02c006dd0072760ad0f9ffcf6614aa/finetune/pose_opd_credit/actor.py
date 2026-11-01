"""Ephemeral two-device actor construction and current-student publication.

Publication is collective on every learner rank, including ranks without an
actor. Cast-before-gather avoids gathering full FP32 master parameters. No
checkpoint is reread, so every rollout uses the immediately preceding update.
"""
from __future__ import annotations

import threading
import time

import torch
import torch.distributed as dist

_BUILD_LOCK = threading.RLock()


def release_publication_cache(devices):
    """Return unused gather allocations to CUDA before the paired actor grows.

    Every learner process must call this: another process cannot reuse its
    peer's cached allocator blocks. Synchronization completes publications and
    cross-device copies first; empty_cache never releases live tensor storage.
    CPU-only callers are a no-op. The caller supplies any required rank barrier.
    """
    started = time.perf_counter()
    cuda_devices = set()
    for value in devices:
        device = torch.device(value)
        if device.type == 'cuda':
            cuda_devices.add(torch.device('cuda', torch.cuda.current_device()) if device.index is None else device)
    cuda_devices = sorted(cuda_devices, key=lambda device: device.index)
    for device in cuda_devices:
        torch.cuda.synchronize(device)
    before = {device: (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
              for device in cuda_devices}
    for device in cuda_devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    result = []
    for device in cuda_devices:
        allocated_before, reserved_before = before[device]
        allocated_after, reserved_after = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
        result.append({'device': str(device), 'allocated_before_gib': allocated_before / 2 ** 30,
                       'allocated_after_gib': allocated_after / 2 ** 30,
                       'reserved_before_gib': reserved_before / 2 ** 30,
                       'reserved_after_gib': reserved_after / 2 ** 30,
                       'released_reserved_gib': (reserved_before - reserved_after) / 2 ** 30})
    return {'seconds': time.perf_counter() - started, 'devices': result}


def _canonical(name):
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


def _tensors(model):
    result = {}
    for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in values:
            name = _canonical(name)
            if name in result:
                raise ValueError(f"Duplicate canonical actor tensor: {name}")
            result[name] = (kind, tensor)
    return result


def _consensus(error, *, label):
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    else:
        errors = [error]
    bad = [(rank, value) for rank, value in enumerate(errors) if value]
    if bad:
        raise RuntimeError(f"{label}: " + "; ".join(f"rank {rank}: {value}" for rank, value in bad))


def _publication_dtype(name, kind, source):
    if kind == "buffer" or not source.is_floating_point():
        return source.dtype
    return torch.float32 if name == "vgain.theta" else torch.bfloat16


def build_actor(student, cfgs, devices, vadapter=None, alpha=1.0, *, backbone_cut=None):
    """Build a bare BF16 PPM over two devices; call publish_actor before using it.

    This runs only on actor-owning ranks. The class is recovered from the
    learner's MRO rather than importing another incompatible model_ppm module.
    Buffers retain learner dtypes; vgain.theta retains FP32 precision.
    """
    started = time.perf_counter()
    devices = tuple(torch.device(device) for device in devices)
    if len(devices) != 2:
        raise ValueError("A pose actor requires exactly two devices")
    ppm = next((cls for cls in type(student).__mro__ if cls.__name__ == "PPM"), None)
    if ppm is None:
        raise TypeError("Student must be the production PPM model")
    c32, c4, c06 = cfgs
    proj_r = int(student.head.proj.G.shape[0])
    tower_vocab = int(student.tower_s.text_emb.num_embeddings)
    has_pose = hasattr(student, "tower_p")
    if not has_pose:
        raise ValueError("The student actor must generate pose")
    if hasattr(student, "vadapter") and not vadapter:
        raise ValueError("Voxtral student actor requires its adapter configuration")
    # Constructors in the archived PPM code allocate with the default dtype.
    with _BUILD_LOCK:
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                model = ppm(c32, c4, c06, proj_r=proj_r, tower_vocab=tower_vocab,
                            pose_head=hasattr(student, "pose_head"),
                            decp_bb=hasattr(student, "decp_bb"), vadapter=vadapter, nopose=False)
        finally:
            torch.set_default_dtype(previous_dtype)
    n_layers = len(model.layers)
    if backbone_cut is not None and (type(backbone_cut) is not int or not 1 <= backbone_cut < n_layers):
        raise ValueError(f"backbone_cut must leave at least one layer on each device (1..{n_layers - 1})")
    # A placement override changes no model arithmetic. The default retains
    # the production 1 : .75 split; a balanced count reduces peak actor memory.
    if backbone_cut is None:
        layer_devices = [devices[0] if i + 0.5 < n_layers / 1.75 else devices[1] for i in range(n_layers)]
    else:
        layer_devices = [devices[0] if i < backbone_cut else devices[1] for i in range(n_layers)]
    first, last = devices
    model.inp.to_empty(device=first)
    for layer, device in zip(model.layers, layer_devices):
        layer.to_empty(device=device)
    for module in (model.head, model.tower_s, model.dec_s):
        module.to_empty(device=last)
    for module in (model.tower_p, model.dec_p):
        module.to_empty(device=first)
    for name in ("decp_bb", "pose_head", "vadapter", "vgain"):
        if hasattr(model, name):
            getattr(model, name).to_empty(device=first)
    if hasattr(model, "vgain"):
        model.vgain.float()
    if hasattr(model, "decp_bb"):
        model.decp_bb_alpha = float(alpha)
    for decoder, device in ((model.dec_s, last), (model.dec_p, first)):
        rotary = decoder.model.rotary_emb
        decoder.model.rotary_emb = type(rotary)(config=c06, device=device)
    source = _tensors(student)
    for name, (kind, target) in _tensors(model).items():
        if kind != "buffer" or name not in source:
            continue
        wanted = source[name][1].dtype
        if target.dtype != wanted:
            owner_name, _, leaf = name.rpartition(".")
            owner = model.get_submodule(owner_name) if owner_name else model
            setattr(owner, leaf, torch.empty(target.shape, dtype=wanted, device=target.device))
    model.requires_grad_(False)
    model.eval()
    model._opd_actor_published = False
    model._opd_build_seconds = time.perf_counter() - started
    model._opd_backbone_cut = (backbone_cut if backbone_cut is not None
                               else sum(i + 0.5 < n_layers / 1.75 for i in range(n_layers)))
    return model, layer_devices


@torch.no_grad()
def publish_actor(student, actor_or_none):
    """Collectively publish all student parameters and buffers to each actor.

    Global manifests and destination layouts are checked before tensor
    collectives. Copy failures are accumulated while every rank completes the
    gather sequence, then agreed once. Allocation/communication failures remain
    fatal, as with FSDP collectives. Avoid per-parameter object barriers.
    """
    started = time.perf_counter()
    error, source, destination = None, {}, {}
    try:
        source = _tensors(student)
        destination = {} if actor_or_none is None else _tensors(actor_or_none)
        if actor_or_none is not None:
            if set(source) != set(destination):
                missing = sorted(set(source) - set(destination))[:5]
                extra = sorted(set(destination) - set(source))[:5]
                raise ValueError(f"Actor tensor keys differ: missing={missing}, extra={extra}")
            for name, (kind, value) in source.items():
                dest_kind, target = destination[name]
                if kind != dest_kind or value.shape != target.shape:
                    raise ValueError(f"Actor tensor shape/kind differs for {name}")
                wanted = _publication_dtype(name, kind, value)
                if target.dtype != wanted:
                    raise ValueError(f"Actor tensor {name} dtype {target.dtype} differs from required {wanted}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _consensus(error, label="Actor publication preflight failed")
    names = sorted(source)
    manifest = [(name, source[name][0], tuple(source[name][1].shape), str(source[name][1].dtype)) for name in names]
    if dist.is_available() and dist.is_initialized():
        manifests = [None] * dist.get_world_size()
        dist.all_gather_object(manifests, manifest)
        if any(other != manifest for other in manifests):
            raise RuntimeError("Learner ranks disagree on the publication tensor manifest")
    published, copy_error = 0, None
    for name in names:
        kind, value = source[name]
        converted = value.detach().to(dtype=_publication_dtype(name, kind, value))
        full = converted.full_tensor() if hasattr(converted, "full_tensor") else converted
        try:
            if actor_or_none is not None and copy_error is None:
                target = destination[name][1]
                target.copy_(full.to(device=target.device), non_blocking=False)
                published += target.numel() * target.element_size()
        except Exception as exc:
            copy_error = f"{name}: {type(exc).__name__}: {exc}"
        del converted, full
    _consensus(copy_error, label="Actor publication copy failed")
    if actor_or_none is not None:
        for device in {tensor.device for _, tensor in destination.values() if tensor.is_cuda}:
            torch.cuda.synchronize(device)
        actor_or_none._opd_actor_published = True
    return {"tensors": len(names), "actor_present": actor_or_none is not None, "copied_bytes": published,
            "elapsed_seconds": time.perf_counter() - started}

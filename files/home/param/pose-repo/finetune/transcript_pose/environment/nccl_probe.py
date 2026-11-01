"""Bounded NCCL/FSDP2 smoke test; no checkpoint, dataset, or optimizer."""
from __future__ import annotations

import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
rank, world = dist.get_rank(), dist.get_world_size()

def announce(stage: str) -> None:
    if rank == 0:
        print(json.dumps({"stage": stage, "world_size": world}), flush=True)

try:
    announce("initialized")
    scalar = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(scalar)
    expected_sum = world * (world + 1) / 2
    assert scalar.item() == expected_sum, (rank, scalar.item(), expected_sum)
    announce("scalar_all_reduce_passed")

    nbytes = 64 * 1024 * 1024
    payload = torch.full((nbytes // 4,), float(rank + 1), device=device)
    dist.all_reduce(payload)
    assert bool((payload == expected_sum).all())
    payload.zero_()
    for _ in range(5):
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
    all_elapsed = [torch.zeros_like(elapsed_tensor) for _ in range(world)]
    dist.all_gather(all_elapsed, elapsed_tensor)
    durations = sorted(t.item() for t in all_elapsed)
    seconds_per_collective = max(durations) / iterations
    alg_gbps = nbytes / seconds_per_collective / 1e9
    bench = {
        "bytes_per_rank": nbytes,
        "iterations": iterations,
        "slowest_rank_ms_per_all_reduce": seconds_per_collective * 1000,
        "fastest_rank_ms_per_all_reduce": min(durations) / iterations * 1000,
        "algorithmic_GB_per_s": alg_gbps,
        "normalized_bus_GB_per_s": alg_gbps * 2 * (world - 1) / world,
    }
    del payload
    announce("64MiB_all_reduce_passed")

    def fsdp_check(mesh_shape: tuple[int, ...], mesh_names: tuple[str, ...]) -> dict:
        mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_names)
        torch.manual_seed(1107)
        model = nn.Sequential(nn.Linear(224, 224), nn.GELU(), nn.Linear(224, 112)).to(device)
        reference = copy.deepcopy(model)
        for layer in (model[0], model[2]):
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        torch.manual_seed(2000 + rank)
        x = torch.randn(8, 224, device=device)
        target = torch.randn(8, 112, device=device)
        output = model(x)
        reference_output = reference(x)
        torch.testing.assert_close(output, reference_output, atol=1e-6, rtol=1e-5)
        (output - target).square().mean().backward()
        (reference_output - target).square().mean().backward()
        max_error = torch.zeros((), device=device)
        shard_numel, full_numel = 0, 0
        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == ref_name
            dist.all_reduce(ref_param.grad)
            ref_param.grad.div_(world)
            full_grad = param.grad.full_tensor()
            torch.testing.assert_close(full_grad, ref_param.grad, atol=2e-7, rtol=2e-5)
            max_error = torch.maximum(max_error, (full_grad - ref_param.grad).abs().max())
            shard_numel += param.to_local().numel()
            full_numel += param.numel()
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
        result = {
            "mesh_shape": list(mesh_shape),
            "mesh_dim_names": list(mesh_names),
            "forward_and_global_mean_gradient_comparison": "passed",
            "max_gradient_abs_error": max_error.item(),
            "rank0_local_parameter_numel": shard_numel,
            "full_parameter_numel": full_numel,
        }
        del model, reference
        dist.barrier()
        return result

    checks = [fsdp_check((world,), ("shard",))]
    announce("fsdp2_world_sharding_passed")
    if world % 8 == 0:
        checks.append(fsdp_check((world // 8, 8), ("replicate", "shard")))
        announce("fsdp2_hybrid_sharding_passed")

    if rank == 0:
        report = {
            "status": "passed",
            "world_size": world,
            "nodes": world // int(os.environ["LOCAL_WORLD_SIZE"]),
            "torch": torch.__version__,
            "nccl_version": list(torch.cuda.nccl.version()),
            "network_environment": {name: os.environ.get(name) for name in (
                "NCCL_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY", "NCCL_NET",
                "NCCL_IB_DISABLE", "NCCL_IB_HCA", "GLOO_SOCKET_IFNAME",
            )},
            "all_reduce": bench,
            "fsdp2": checks,
        }
        report_path = Path(os.environ["POSE_NCCL_PROBE_OUTPUT"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    dist.barrier()
finally:
    dist.destroy_process_group()

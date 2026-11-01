import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

lock = open('/home/param/runs/pose-runs/.duplex-distill-gpus.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise RuntimeError('Another GPU process is present; refusing the isolated microbenchmark')
started = time.time()
source = Path('/home/param/pose-worktrees/opd-bench-ladder-cfg3-v3/infer/engine/ppm/light_kernels.py')
sys.path.insert(0, str(source.parent))
import torch
import torch.nn.functional as F
import light_kernels as LK

torch.set_num_threads(1)
torch.manual_seed(193)
rows = []
for inputs, outputs in ((1024, 2048), (1024, 6144), (3072, 1024)):
    # Every graph traverses at least128MiB of distinct weights, exceeding H100 L2.
    weights = [torch.randn(outputs, inputs, dtype=torch.bfloat16, device='cuda').mul_(.02) for _ in range(32)]
    all_x = torch.randn(8, inputs, dtype=torch.bfloat16, device='cuda')
    for batch in (4, 8):
        x = all_x[:batch].contiguous()
        numerics = []
        for w in weights:
            native, reference = LK.gemv(x, w), F.linear(x, w)
            difference = native.float() - reference.float()
            numerics.append(dict(max_abs=float(difference.abs().max()),
                mean_abs=float(difference.abs().mean()), unequal=int((native != reference).sum()), count=native.numel()))
        for implementation, function in (('native_gemv', LK.gemv), ('torch_linear', F.linear)):
            for _ in range(3):
                ys = [function(x, w) for w in weights]
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ys = [function(x, w) for w in weights]
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            observations = []
            for _ in range(7):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                observations.append(begin.elapsed_time(end) * 1000 / (20 * len(weights)))
            rows.append(dict(input_dim=inputs, output_dim=outputs, batch=batch,
                implementation=implementation, native_dispatch='CUDA-core _gemv_small_k' if batch <= 4 else 'tensor-core _gemv_k',
                distinct_weights=len(weights), weight_working_set_bytes=sum(w.numel() * w.element_size() for w in weights),
                latency_us=dict(median=statistics.median(observations), min=min(observations), max=max(observations)),
                vs_bf16_torch_linear=dict(max_abs=max(n['max_abs'] for n in numerics),
                    mean_abs=statistics.mean(n['mean_abs'] for n in numerics),
                    unequal_fraction=sum(n['unequal'] for n in numerics) / sum(n['count'] for n in numerics))))
            del graph, ys
        del x
    del weights, all_x
report = dict(host=os.uname().nodename, device=torch.cuda.get_device_name(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    torch_version=torch.__version__, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    ppm_autotune=os.environ.get('PPM_AUTOTUNE'), elapsed_seconds=time.time()-started,
    notes=['CUDA graph replay amortizes Python launch overhead.', '32 distinct weight matrices per graph; minimum128MiB exceeds H100 L2.',
           'BF16 inputs/weights/output; numerical comparison is native versus torch BF16 F.linear.',
           'Synthetic layer microbenchmark only; not full decoder or training throughput.'], rows=rows)
destination = Path('/home/param/outputs/pose-opd-credit-bench/gemv-m4-m8.json')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

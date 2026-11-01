# wan-vae-collection@588eaeb utils/raster.py (trimmed: imports made package-relative, venv bin put on PATH for ninja)
"""
Skeleton canvas rasterizer: pose_recon [B, T, 165] + global_state [B, T, 3]
-> canvas [B, 3, T, H, W] in [-1, 1] on GPU, via the CUDA kernel in
csrc/pose_raster.cu (JIT-compiled on first use, ~1 min, needs ninja).
Bones and joints are drawn with a confidence colour ramp (red -> yellow ->
green over [conf_min, conf_hi], fading out below conf_min to fade_floor).
RENDER is the aloe production canvas.
"""

import os
import sys
from pathlib import Path

import torch

from .kinematic import (BODY_CONNECTIONS, FACE_CONNECTIONS,
                        LEFT_HAND_CONNECTIONS, NUM_JOINTS,
                        RIGHT_HAND_CONNECTIONS, offsets_to_positions)

CONNECTIONS = (FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS
               + RIGHT_HAND_CONNECTIONS)
JOINTS = sorted({k for pair in CONNECTIONS for k in pair})
# part group per primitive: 0 face, 1 body, 2 left hand, 3 right hand
CONN_GROUPS = ([0] * len(FACE_CONNECTIONS) + [1] * len(BODY_CONNECTIONS)
               + [2] * len(LEFT_HAND_CONNECTIONS)
               + [3] * len(RIGHT_HAND_CONNECTIONS))
JOINT_GROUPS = [0 if j < 5 else 1 if j < 13 else 2 if j < 34 else 3
                for j in JOINTS]

RENDER = dict(line_w=2.0, dot_r=2.0, conf_min=0.2, conf_hi=0.6,
              fade_floor=0.05)

_CU = Path(__file__).resolve().parent / "csrc" / "pose_raster.cu"
_CPP = ("#include <torch/extension.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n"
        "void render_launch(const float *py, const float *px, const float *conf, float *out, int B, int T, int Hh, "
        "int Ww, double line_thresh, double dot_thresh, double cmin, double chi, double ffloor, bool part_colors, "
        "cudaStream_t stream);\n"
        "void render(torch::Tensor py, torch::Tensor px, torch::Tensor conf, "
        "torch::Tensor out, double line_thresh, double dot_thresh, "
        "double cmin, double chi, double ffloor, bool part_colors) {\n"
        "  render_launch(py.data_ptr<float>(), px.data_ptr<float>(), conf.data_ptr<float>(), out.data_ptr<float>(), "
        "py.size(0), py.size(1), out.size(3), out.size(4), line_thresh, dot_thresh, cmin, chi, ffloor, part_colors, "
        "at::cuda::getCurrentCUDAStream());\n"
        "}\n")  # edit (t2v): the torch wrapper lives here (g++), the .cu has the kernel + a raw-pointer launcher
_ext = None


def get_raster_ext():
    global _ext
    if _ext is None:
        os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH']
        cu = (_CU.read_text()
              .replace("__NCONN__", str(len(CONNECTIONS)))
              .replace("__NJOINT__", str(len(JOINTS)))
              .replace("__NUMJ__", str(NUM_JOINTS))
              .replace("__CONN_INIT__",
                       ",".join(f"{{{i},{j}}}" for i, j in CONNECTIONS))
              .replace("__JOINT_INIT__", ",".join(map(str, JOINTS)))
              .replace("__CONN_GROUP_INIT__", ",".join(map(str, CONN_GROUPS)))
              .replace("__JOINT_GROUP_INIT__",
                       ",".join(map(str, JOINT_GROUPS))))
        # edit (t2v): load() on written files instead of load_inline, which prepends <torch/types.h> to the CUDA source
        # (nvcc 13.3 + gcc 11 cannot parse torch 2.11's List_inl.h); kernel and wrapper are unchanged in substance
        import hashlib
        from torch.utils.cpp_extension import load
        cpp = _CPP + ("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"render\", &render, \"render\"); }\n")
        tag = hashlib.sha1((cpp + cu).encode()).hexdigest()[:10]
        src = Path(os.path.expanduser('~/.cache/torch_extensions/pose_raster_src')) / tag
        src.mkdir(parents=True, exist_ok=True)
        (src / 'wrap.cpp').write_text(cpp)
        (src / 'raster.cu').write_text(cu)
        _ext = load(name=f"pose_raster_fp32_t2v_{tag}", sources=[str(src / 'wrap.cpp'), str(src / 'raster.cu')],
                    extra_cuda_cflags=["-O3"], verbose=False)
    return _ext


@torch.library.custom_op("pose_raster::render_canvas", mutates_args=())
def _render(pose_recon: torch.Tensor, global_state: torch.Tensor,
            canvas_h: int, canvas_w: int, line_w: float, dot_r: float,
            conf_min: float, conf_hi: float, fade_floor: float) -> torch.Tensor:
    B, T, _ = pose_recon.shape
    feat = pose_recon.view(B, T, NUM_JOINTS, 3).float()
    offsets, conf = feat[..., :2], feat[..., 2]
    root = global_state[..., :2].float()
    # one shoulder width per clip, taken from its first frame so no frame's
    # canvas depends on later frames
    msw = global_state[:, 0, 2].float()
    pos = offsets_to_positions(root, offsets * msw[:, None, None, None])
    py = (pos[..., 0] * canvas_h).contiguous()
    px = (pos[..., 1] * canvas_w).contiguous()
    out = torch.empty(B, 3, T, canvas_h, canvas_w, device=pose_recon.device,
                      dtype=torch.float32)
    get_raster_ext().render(py, px, conf.contiguous(), out, line_w / 2.0,
                            max(1.0, dot_r), conf_min, conf_hi, fade_floor,
                            False)
    return out


@_render.register_fake
def _(pose_recon, global_state, canvas_h, canvas_w, line_w, dot_r, conf_min,
      conf_hi, fade_floor):
    B, T, _ = pose_recon.shape
    return pose_recon.new_empty(B, 3, T, canvas_h, canvas_w,
                                dtype=torch.float32)


@torch.no_grad()
def render_pose_canvas(pose_recon, global_state, canvas_h, canvas_w,
                       **render):
    """[B, T, 165] + [B, T, 3] (cuda) -> [B, 3, T, canvas_h, canvas_w] in [-1, 1]."""
    return _render(pose_recon, global_state, canvas_h, canvas_w,
                   **{**RENDER, **render})

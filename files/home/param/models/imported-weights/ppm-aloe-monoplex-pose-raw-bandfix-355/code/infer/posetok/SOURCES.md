## pose/ — posev3 pose tokens → electron v3 pose latents

Pipeline (as the v2pure sample folders were made by `assemble.py build` → `ev2win.window` → `ev2.py`):
PoseTokenizer decode (TF32 off) → denormalise against frame 0 → `raw_to_recon` → `render_pose_canvas` 480x256 →
`FastElectronEncoder(encoder-v2-pose)` (bf16, chunk 32, `torch.compile` default mode, `cudnn.benchmark=True`), last frame repeated once (2F+1 frames).

Weights (HF, pinned, default token):

| weight | repo @ revision | file |
|---|---|---|
| electron encoder | `vanshjjw/electron-vae` @ `9f18375922780ad10fbc209af78c467a67866e69` | `encoder-v2-pose.safetensors` (sha256 `5f73f266…`) |
| pose tokenizer | `InternalCan/pose-tokenizer-32cb-200k-noshift` @ `1d6ff4f0de89408cb054da618e309d4334596252` | snapshot (`model.safetensors` sha256 `8430bc8c…`, config) |

Vendored code (each file's first line names its upstream):

| vendored file | upstream (repo@rev path) | trimmed |
|---|---|---|
| `pose/__init__.py` | new; `ev2.py` `load_encoder`/`decode`/`to_px`/`refs_raw_to_recon` (gs only)/`v1_path` + `ev2win.window` inlined | `load()`, `tokens_to_latents()`, odd F trimmed to even |
| `pose/__main__.py` | new | CLI `python -m pose POSE FRAME0 OUT` |
| `pose/vae.py` | `wan-vae-collection@588eaeb models/muon/vae.py` | only `CACHE_T`, `CausalConv3d`, `RMS_norm`, `ResidualBlock` (init only), `AttentionBlock`, `patchify`, `AvgDown3D`; streaming `feat_cache` paths, Resample/Up*/Encoder3d/Decoder3d dropped |
| `pose/encoder.py` | `wan-vae-collection@588eaeb models/electron/encoder.py` | `_DownBlock`/`ElectronEncoder` modules only; `forward` (checkpointed training path) dropped |
| `pose/fast_encoder.py` | `wan-vae-collection@588eaeb optimisations/encoder.py` | `_FastEncoder._finish/_run/encode`, `_FastStudent`, `FastElectronEncoder`; Muon/Proton/Kaon/TAE encoders and `DTYPE_MAP` dropped (dtype passed as `torch.bfloat16`) |
| `pose/kinematic.py` | `wan-vae-collection@588eaeb utils/kinematic.py` | verbatim |
| `pose/raster.py` | `wan-vae-collection@588eaeb utils/raster.py` | package-relative import; venv `bin/` prepended to PATH (ninja) inside `get_raster_ext` |
| `pose/csrc/pose_raster.cu` | `wan-vae-collection@588eaeb utils/csrc/pose_raster.cu` | verbatim (CUDA kernel, JIT-built via `load_inline`, C++ binding declared in `raster.py`) |
| `pose/pose_tokenizer/__init__.py` | `HF InternalCan/sam-pose-aloe-v2pure-step636@cbd9c1e3 pose_tokenizer/__init__.py` | `defaults` import dropped |
| `pose/pose_tokenizer/config.py`, `tokenizer.py` | same repo, `pose_tokenizer/config.py`, `tokenizer.py` | verbatim |
| `pose/pose_tokenizer/model/{__init__,layers,vq,encoder,decoder,seanet_transformer,tokenizer}.py` | same repo, `pose_tokenizer/model/*` | verbatim |
| `pose/pose_tokenizer/data/__init__.py` | same repo, `pose_tokenizer/data/__init__.py` | emptied: no dataset/rendering/visualization imports, so no cv2 (stub no longer needed) |
| `pose/pose_tokenizer/data/kinematic.py` | same repo, `pose_tokenizer/data/kinematic.py` | verbatim |

The HF-vendored `pose_tokenizer` is code-identical to the tokenizer-lib the samples were made with (`model/`, `tokenizer.py`, `config.py`, `data/kinematic.py` byte-identical; only `data/__init__.py` imports, `defaults.py` and `training/` differ, none used by decode).

Reproducibility vs `v2pure/samples/<n>/pose_latents_electron_v3.safetensors` (m7 GPU 6/7):
- Decode, frame-0 global state, keypoints, `raw_to_recon` and the canvas are bit-identical to the original `ev2win`/`ev2` code; with `use_compile=False` and deterministic cudnn, the vendored encoder is bit-identical to the original encoder.
- The original pipeline itself is not run-to-run deterministic: `cudnn.benchmark=True` (set in `FastElectronEncoder`) picks conv algorithms by timing, and inductor autotunes Triton kernels at runtime. Re-running the unmodified `ev2win.window` path in fresh processes gives rel 0.0011–0.0022, max 0.03125 (1 bf16 ulp) vs the samples, different on every run, with the default or a fresh inductor cache. The samples can't be reproduced bit-exactly by any implementation.
- `pose/` (compiled, benchmark on, same as the original): samples 1/2/3 rel 0.0019/0.0022/0.0014, max 0.03125, `n_frames` 301 / `valid_frames` 300 int64 match. That is within the original's own rerun spread.

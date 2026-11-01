// wan-vae-collection@588eaeb utils/csrc/pose_raster.cu
// pose_raster.cu — FP32 single-pass pose score-canvas rasterizer (CUDA).
//
// JIT-compiled by utils/raster.py via torch.utils.cpp_extension.load_inline.
// One CUDA thread owns one output pixel and loops over all 107 primitives
// (skeleton topology baked into __constant__ memory), keeping the running
// color in registers (last-match-wins) and writing the [B,3,T,H,W] canvas
// exactly once.
//
// TEMPLATE: the __NCONN__, __NJOINT__, __NUMJ__, __*_INIT__ tokens are
// substituted at load time from utils/kinematic.py, so this file is not
// standalone-compilable.

// edit (t2v): no torch headers in the CUDA translation unit (nvcc 13.3 + gcc 11 cannot parse torch 2.11's
// List_inl.h); the torch::Tensor wrapper moved to the C++ source in raster.py, the kernel is unchanged.
#include <cuda_runtime.h>

#define NCONN  __NCONN__
#define NJOINT __NJOINT__
#define NJ     __NUMJ__

__constant__ int CONN[NCONN][2]   = { __CONN_INIT__ };
__constant__ int JOINTS_ARR[NJOINT] = { __JOINT_INIT__ };
// group per primitive: 0=face 1=body 2=left hand 3=right hand
__constant__ int CONN_GROUP[NCONN]    = { __CONN_GROUP_INIT__ };
__constant__ int JOINT_GROUP[NJOINT]  = { __JOINT_GROUP_INIT__ };

__device__ __forceinline__ void conf_to_color(
        float c, int grp, int part_colors,
        float cmin, float chi, float ffloor, float &R, float &G, float &B) {
    c = fminf(fmaxf(c, 0.f), 1.f);
    float t = (c - cmin) / (chi - cmin);
    t = fminf(fmaxf(t, 0.f), 1.f);
    float fade = (c - ffloor) / (cmin - ffloor);
    fade = fminf(fmaxf(fade, 0.f), 1.f);
    if (part_colors && grp >= 2) {
        // Hands share the low-confidence red anchor with the body ramp, then
        // diverge with confidence toward part-identity hues. Above t=0 a hand
        // always has B>0 (body has B=0); left keeps G=0, right has G>0.
        if (grp == 2) {                    // left: red -> rose -> magenta
            R = 255.f; G = 0.f; B = 255.f * t;
        } else if (t < 0.5f) {             // right: red -> pink/white ...
            R = 255.f; G = B = 255.f * (t / 0.5f);
        } else {                           //        ... -> cyan
            R = 255.f * (1.f - (t - 0.5f) / 0.5f); G = B = 255.f;
        }
        R *= fade; G *= fade; B *= fade;
        return;
    }
    // Face/body: original confidence ramp (red->yellow->green), unchanged.
    float r = (t < 0.5f) ? 255.f : 255.f * (1.f - (t - 0.5f) / 0.5f);
    float g = (t < 0.5f) ? 255.f * (t / 0.5f) : 255.f;
    R = r * fade;
    G = g * fade;
    B = 0.f;
}

// one thread per output pixel; one block = one (b,t) frame's scanline of W pixels.
__global__ void raster_kernel(
        const float *__restrict__ py, const float *__restrict__ px,
        const float *__restrict__ conf, float *__restrict__ out,
        int B, int T, int Hh, int Ww,
        float line_thresh, float dot_thresh, float cmin, float chi, float ffloor,
        int part_colors) {
    int bt = blockIdx.z;            // b*T + t
    int b  = bt / T;
    int t  = bt - b * T;
    int y  = blockIdx.y;
    int x  = blockIdx.x * blockDim.x + threadIdx.x;

    // frame's 55 joints into shared mem (reused by all W threads in the block)
    __shared__ float s_py[NJ], s_px[NJ], s_conf[NJ];
    for (int j = threadIdx.x; j < NJ; j += blockDim.x) {
        int base = (b * T + t) * NJ + j;
        s_py[j] = py[base]; s_px[j] = px[base]; s_conf[j] = conf[base];
    }
    __syncthreads();
    if (x >= Ww) return;

    float fy = (float)y, fx = (float)x;
    float cr = 0.f, cg = 0.f, cb = 0.f;   // running canvas color

    // line segments: point-to-segment distance, last match wins
    for (int k = 0; k < NCONN; ++k) {
        int i = CONN[k][0], j = CONN[k][1];
        float y1 = s_py[i], x1 = s_px[i];
        float ay = s_py[j] - y1, ax = s_px[j] - x1;
        float L2 = fmaxf(ay * ay + ax * ax, 1e-6f);
        float wy = fy - y1, wx = fx - x1;
        float tt = fminf(fmaxf((wy * ay + wx * ax) / L2, 0.f), 1.f);
        float dy = wy - tt * ay, dx = wx - tt * ax;
        float d = sqrtf(dy * dy + dx * dx);
        float c = fminf(s_conf[i], s_conf[j]);
        if (d <= line_thresh && c >= ffloor)
            conf_to_color(c, CONN_GROUP[k], part_colors, cmin, chi, ffloor, cr, cg, cb);
    }
    // joints: point distance
    for (int jj = 0; jj < NJOINT; ++jj) {
        int j = JOINTS_ARR[jj];
        float dy = fy - s_py[j], dx = fx - s_px[j];
        float d = sqrtf(dy * dy + dx * dx);
        float c = s_conf[j];
        if (d <= dot_thresh && c >= ffloor)
            conf_to_color(c, JOINT_GROUP[jj], part_colors, cmin, chi, ffloor, cr, cg, cb);
    }

    // write [B,3,T,H,W] once; value = canvas/255*2 - 1
    long HW = (long)Hh * Ww;
    long sc = (long)T * HW;                       // stride between channel planes
    long p  = ((long)(b * 3) * T + t) * HW + (long)y * Ww + x;   // R plane (c=0)
    out[p]          = cr / 255.f * 2.f - 1.f;     // R
    out[p + sc]     = cg / 255.f * 2.f - 1.f;     // G
    out[p + 2 * sc] = cb / 255.f * 2.f - 1.f;     // B (0 unless part_colors)
}

void render_launch(const float *py, const float *px, const float *conf, float *out,
                   int B, int T, int Hh, int Ww,
                   double line_thresh, double dot_thresh, double cmin, double chi, double ffloor,
                   bool part_colors, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((Ww + block.x - 1) / block.x, Hh, B * T);
    raster_kernel<<<grid, block, 0, stream>>>(
        py, px, conf, out, B, T, Hh, Ww,
        (float)line_thresh, (float)dot_thresh, (float)cmin, (float)chi, (float)ffloor,
        part_colors ? 1 : 0);
}

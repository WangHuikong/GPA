#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace mxfp8_gemv {

constexpr int kMxfp8K = 4096;
constexpr int kMxfp8N = 4096;
constexpr int kMxfp8GroupSize = 32;
constexpr int kMxfp8Groups = kMxfp8K / kMxfp8GroupSize;
constexpr int kE4M3Bias = 7;
constexpr int kE8M0Bias = 127;

static_assert(kMxfp8K % kMxfp8GroupSize == 0, "K must be multiple of group size.");

__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t v) {
  const int sign = (v >> 7) & 0x1;
  const int exp = (v >> 3) & 0xF;
  const int mant = v & 0x7;

  if (exp == 0) {
    if (mant == 0) {
      return sign ? -0.0f : 0.0f;
    }
    const float frac = static_cast<float>(mant) * (1.0f / 8.0f);
    const float val = ldexpf(frac, 1 - kE4M3Bias);
    return sign ? -val : val;
  }

  if (exp == 0xF) {
    if (mant == 0) {
      return sign ? -CUDART_INF_F : CUDART_INF_F;
    }
    return CUDART_NAN_F;
  }

  const float frac = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  const float val = ldexpf(frac, exp - kE4M3Bias);
  return sign ? -val : val;
}

__device__ __forceinline__ float fp8_e8m0_to_f32(uint8_t v) {
  if (v == 0) {
    return 0.0f;
  }
  if (v == 0xFF) {
    return CUDART_INF_F;
  }
  return ldexpf(1.0f, static_cast<int>(v) - kE8M0Bias);
}

// Layout: A[4096], scale_A[128], B[4096 x 4096], scale_B[128 x 4096].
__global__ void mxfp8_gemv_kernel(const uint8_t* a,
                                  const uint8_t* scale_a,
                                  const uint8_t* b,
                                  const uint8_t* scale_b,
                                  float* out) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= kMxfp8N) {
    return;
  }

  float acc = 0.0f;
  for (int group = 0; group < kMxfp8Groups; ++group) {
    const float scale_a_f = fp8_e8m0_to_f32(scale_a[group]);
    const float scale_b_f = fp8_e8m0_to_f32(scale_b[group * kMxfp8N + col]);
    const int base = group * kMxfp8GroupSize;

#pragma unroll
    for (int i = 0; i < kMxfp8GroupSize; ++i) {
      const int k = base + i;
      const float a_f = fp8_e4m3_to_f32(a[k]) * scale_a_f;
      const float b_f = fp8_e4m3_to_f32(b[k * kMxfp8N + col]) * scale_b_f;
      acc += a_f * b_f;
    }
  }

  out[col] = acc;
}

cudaError_t launch_mxfp8_gemv(const uint8_t* a,
                              const uint8_t* scale_a,
                              const uint8_t* b,
                              const uint8_t* scale_b,
                              float* out,
                              cudaStream_t stream) {
  const dim3 block(256);
  const dim3 grid((kMxfp8N + block.x - 1) / block.x);
  mxfp8_gemv_kernel<<<grid, block, 0, stream>>>(a, scale_a, b, scale_b, out);
  return cudaGetLastError();
}

}  // namespace mxfp8_gemv

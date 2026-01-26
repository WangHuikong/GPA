#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Simple software FP8 (E4M3) encode/decode used for demonstration.
// - 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits
// - Subnormals are treated as zero
// - Values are saturated to the max representable magnitude

struct fp8_e4m3 {
  uint8_t data;
};

__host__ __device__ inline float fp8_max_value() { return 240.0f; }

__host__ __device__ inline fp8_e4m3 fp8_from_float(float x) {
  fp8_e4m3 out{0};
  if (x == 0.0f) {
    return out;
  }

  const int exp_bias = 7;
  const int exp_max = 7;
  const int exp_min = -6;
  const float min_norm = ldexpf(1.0f, exp_min);
  const float max_val = fp8_max_value();

  int sign = x < 0.0f ? 1 : 0;
  float ax = fabsf(x);
  if (!isfinite(ax)) {
    ax = max_val;
  }
  if (ax < min_norm) {
    return out;
  }
  if (ax > max_val) {
    ax = max_val;
  }

  int exp;
  float mant = frexpf(ax, &exp);  // ax = mant * 2^exp, mant in [0.5, 1)
  mant *= 2.0f;                   // mant in [1, 2)
  exp -= 1;

  if (exp > exp_max) {
    exp = exp_max;
    mant = 1.875f;  // 1 + 7/8
  } else if (exp < exp_min) {
    return out;
  }

  float frac = mant - 1.0f;
  int mant_bits = static_cast<int>(floorf(frac * 8.0f + 0.5f));
  if (mant_bits >= 8) {
    mant_bits = 0;
    exp += 1;
    if (exp > exp_max) {
      exp = exp_max;
      mant_bits = 7;
    }
  }

  int exp_bits = exp + exp_bias;
  out.data = static_cast<uint8_t>((sign << 7) | ((exp_bits & 0xF) << 3) |
                                  (mant_bits & 0x7));
  return out;
}

__host__ __device__ inline float fp8_to_float(fp8_e4m3 v) {
  if (v.data == 0) {
    return 0.0f;
  }
  int sign = (v.data >> 7) & 0x1;
  int exp_bits = (v.data >> 3) & 0xF;
  int mant_bits = v.data & 0x7;
  if (exp_bits == 0) {
    return 0.0f;
  }

  int exp = exp_bits - 7;
  float mant = 1.0f + static_cast<float>(mant_bits) / 8.0f;
  float val = ldexpf(mant, exp);
  return sign ? -val : val;
}

static void check_cuda(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line,
                 cudaGetErrorString(err));
    std::exit(1);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

__global__ void gemv_fp8_kernel(const fp8_e4m3 *A, const fp8_e4m3 *x,
                                float *y, int M, int N, float scale_a,
                                float scale_x) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) {
    return;
  }

  float acc = 0.0f;
  int base = row * N;
  for (int col = 0; col < N; ++col) {
    float a = fp8_to_float(A[base + col]) * scale_a;
    float b = fp8_to_float(x[col]) * scale_x;
    acc += a * b;
  }
  y[row] = acc;
}

static float compute_scale(const std::vector<float> &data) {
  float max_abs = 0.0f;
  for (float v : data) {
    max_abs = std::max(max_abs, fabsf(v));
  }
  if (max_abs == 0.0f) {
    return 1.0f;
  }
  return max_abs / fp8_max_value();
}

static std::vector<fp8_e4m3> quantize_fp8(const std::vector<float> &data,
                                          float scale) {
  std::vector<fp8_e4m3> out(data.size());
  float inv_scale = (scale == 0.0f) ? 1.0f : 1.0f / scale;
  for (size_t i = 0; i < data.size(); ++i) {
    out[i] = fp8_from_float(data[i] * inv_scale);
  }
  return out;
}

int main(int argc, char **argv) {
  int M = 256;
  int N = 256;
  if (argc >= 3) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
  }
  if (M <= 0 || N <= 0) {
    std::fprintf(stderr, "Invalid M/N: %d %d\n", M, N);
    return 1;
  }

  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> hA(static_cast<size_t>(M) * N);
  std::vector<float> hx(static_cast<size_t>(N));
  for (float &v : hA) {
    v = dist(rng);
  }
  for (float &v : hx) {
    v = dist(rng);
  }

  float scale_a = compute_scale(hA);
  float scale_x = compute_scale(hx);
  std::vector<fp8_e4m3> hA_fp8 = quantize_fp8(hA, scale_a);
  std::vector<fp8_e4m3> hx_fp8 = quantize_fp8(hx, scale_x);

  fp8_e4m3 *dA = nullptr;
  fp8_e4m3 *dx = nullptr;
  float *dy = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, hA_fp8.size() * sizeof(fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&dx, hx_fp8.size() * sizeof(fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&dy, static_cast<size_t>(M) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA_fp8.data(), hA_fp8.size() * sizeof(fp8_e4m3),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dx, hx_fp8.data(), hx_fp8.size() * sizeof(fp8_e4m3),
                        cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  gemv_fp8_kernel<<<blocks, threads>>>(dA, dx, dy, M, N, scale_a, scale_x);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> hy(static_cast<size_t>(M));
  CHECK_CUDA(
      cudaMemcpy(hy.data(), dy, static_cast<size_t>(M) * sizeof(float),
                 cudaMemcpyDeviceToHost));

  std::vector<float> href(static_cast<size_t>(M), 0.0f);
  for (int row = 0; row < M; ++row) {
    float acc = 0.0f;
    int base = row * N;
    for (int col = 0; col < N; ++col) {
      float a = fp8_to_float(hA_fp8[base + col]) * scale_a;
      float b = fp8_to_float(hx_fp8[col]) * scale_x;
      acc += a * b;
    }
    href[row] = acc;
  }

  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  for (int i = 0; i < M; ++i) {
    float diff = fabsf(href[i] - hy[i]);
    max_abs_err = std::max(max_abs_err, diff);
    float denom = fabsf(href[i]);
    if (denom < 1e-6f) {
      denom = 1.0f;
    }
    max_rel_err = std::max(max_rel_err, diff / denom);
  }

  std::printf("FP8 GEMV (E4M3) M=%d N=%d\n", M, N);
  std::printf("scale_a=%.6f scale_x=%.6f\n", scale_a, scale_x);
  std::printf("max_abs_err=%.6e max_rel_err=%.6e\n", max_abs_err, max_rel_err);

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));

  return 0;
}

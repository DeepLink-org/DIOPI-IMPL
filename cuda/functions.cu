/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#include <diopi/functions.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda.h>

#include "helper.hpp"

#define dispatch_dtype(fun, dtype, gridSize, blockSize, stream, ...)                             \
    if (diopi_dtype_int32 == dtype) {                                                            \
        fun<int32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    } else if (diopi_dtype_uint32 == dtype) {                                                    \
        fun<uint32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                          \
    } else if (diopi_dtype_int16 == dtype) {                                                      \
        fun<int16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    } else if (diopi_dtype_uint16 == dtype) {                                                     \
        fun<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                          \
    } else if (diopi_dtype_int8 == dtype) {                                                       \
        fun<int8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                            \
    } else if (diopi_dtype_uint8 == dtype) {                                                      \
        fun<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    } else if (diopi_dtype_float32 == dtype) {                                                    \
        fun<float><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                             \
    } else if (diopi_dtype_float64 == dtype) {                                                    \
        fun<double><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                            \
    } else if (diopi_dtype_bool == dtype) {                                                       \
        fun<bool><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                              \
    } else {                                                                                     \
        fprintf(stderr, "%s:%s: %s<%s %d><<<%d,%d>>>(%s)", __FILE__, __FUNCTION__, #fun, #dtype, \
                dtype, gridSize, blockSize, #__VA_ARGS__);                                       \
        return diopiDtypeNotSupported;                                                           \
    }

template<typename T> __global__
void vecAdd(const void* a, const void* b, void* c, const int numel, const T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    const T* B = static_cast<const T*>(b);
    T* C = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + alpha * B[id];
    }
}

template<typename T> __global__
void vecAddBroadcast(const void* a, const void* b, void* c, const int numel, const T alpha,
        const int64_t* stride1, const int64_t* stride2, const int64_t* outStride, const int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    const T* B = static_cast<const T*>(b);
    T* C = static_cast<T*>(c);
    int size = id;
    size_t idxA = 0;
    size_t idxB = 0;
    if (id < numel) {
        for (int i = 0; i < len; ++i) {
            int tmp = size / outStride[i];
            idxA += tmp * stride1[i];
            idxB += tmp * stride2[i];
            size = size % outStride[i];
        }
        C[id] = A[idxA] + alpha * B[idxB];
    }
}

template<typename T> __global__
void vecAddScalar(const void* a, const T b, void* c, const int numel, const T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    T* C = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + alpha * b;
    }
}

bool compareShape(const diopiSize_t& size1, const diopiSize_t& size2) {
    if (size1.len == size2.len) {
        for (int i = 0; i < size1.len; ++i) {
            if (size1.data[i] != size2.data[i]) {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

void computeStride(const diopiSize_t& size1, const diopiSize_t& size2, diopiSize_t outSize,
        int64_t* stride1, int64_t* stride2) {
    int length = size1.len;
    int len = outSize.len;
    int64_t stride = 1;
    for (int i = 0; i < len; ++i) {
        stride1[i] = 0;
        stride2[i] = 0;
    }
    for (int i = 1; i < length + 1; ++i) {
        if (size1.data[length - i] == outSize.data[len - i]) {
            stride1[len - i] = stride;
            stride *= outSize.data[len - i];
        }
    }
    length = size2.len;
    stride = 1;
    for (int i = 1; i < length + 1; ++i) {
        if (size2.data[length - i] == outSize.data[len - i]) {
            stride2[len - i] = stride;
            stride *= outSize.data[len - i];
        }
    }
}

extern "C" diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    auto stream  = impl::cuda::getStream(ctx);
    auto trInput = impl::cuda::makeTensor(input);
    auto trOther = impl::cuda::makeTensor(other);
    auto trOut   = impl::cuda::makeTensor(out);

    int blockSize = 256;
    double coff = 0.0;
    if (trInput.dtype() <= 7) {
        coff = alpha->ival;
    } else {
        coff = alpha->fval;
    }
    diopiSize_t inShape = trInput.shape();
    diopiSize_t othShape = trOther.shape();
    int gridSize  = (trOut.numel() + blockSize - 1) / blockSize;
    if (compareShape(inShape, othShape)) {
        dispatch_dtype(vecAdd, trInput.dtype(), gridSize, blockSize, stream,
            trInput.data(), trOther.data(), trOut.data(), trInput.numel(), coff);
    } else {
        diopiSize_t outShape = trOut.shape();
        diopiSize_t outStrideHost = trOut.stride();
        int len = outShape.len;
        int64_t nbytes = len * sizeof(int64_t);

        std::vector<int64_t> inStrideHost(len);
        std::vector<int64_t> othStrideHost(len);
        auto inStride = impl::cuda::requiresBuffer(ctx, nbytes);
        auto othStride = impl::cuda::requiresBuffer(ctx, nbytes);
        auto outStride = impl::cuda::requiresBuffer(ctx, nbytes);

        computeStride(inShape, othShape, outShape, inStrideHost.data(), othStrideHost.data());
        cudaMemcpyAsync(inStride.data(), inStrideHost.data(), nbytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(othStride.data(), othStrideHost.data(), nbytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(outStride.data(), outStrideHost.data, nbytes, cudaMemcpyHostToDevice, stream);

        dispatch_dtype(vecAddBroadcast, trInput.dtype(), gridSize, blockSize, stream,
           trInput.data(), trOther.data(), trOut.data(), trOut.numel(), coff, static_cast<const int64_t*>(inStride.data()),
           static_cast<const int64_t*>(othStride.data()), static_cast<const int64_t*>(outStride.data()), len);
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    auto stream  = impl::cuda::getStream(ctx);
    auto trInput = impl::cuda::makeTensor(input);
    auto trOut   = impl::cuda::makeTensor(out);
    int blockSize = 256;
    double coff = 0.0;
    double otherVal = 0.0;
    if (trInput.dtype() <= 7) {
        coff = alpha->ival;
        otherVal = other->ival;
    } else {
        coff = alpha->fval;
        otherVal = other->fval;
    }
    int gridSize = (trInput.numel() + blockSize - 1) / blockSize;
    dispatch_dtype(vecAddScalar, trInput.dtype(), gridSize, blockSize, stream,
        trInput.data(), otherVal, trOut.data(), trInput.numel(), coff);
    return diopiSuccess;
}

template<typename T> __global__
void vecFill(void* a, const float value, const int numel) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    T* A = static_cast<T*>(a);
    if (id < numel) {
        A[id] = static_cast<T>(value);
    }
}

extern "C" diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto stream = impl::cuda::getStream(ctx);
    auto tr = impl::cuda::makeTensor(input);

    diopiDevice_t device = tr.device();
    diopiDtype_t  dtype  = tr.dtype();
    int64_t       numel  = tr.numel();
    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }
    if (diopi_host == device) {
        return diopiErrorOccurred;
    } else {
        int blockSize = 256;
        int gridSize  = (numel + blockSize - 1) / blockSize;
        dispatch_dtype(vecFill, dtype, gridSize, blockSize, stream, tr.data(), val, numel);
    }

    return diopiSuccess;
}


#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
  for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
    for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144

template <typename scalar_t>
__global__ void chamfer_distance_forward_cuda_kernel(int b, int n,
                                                     const scalar_t* xyz, int m,
                                                     const scalar_t* xyz2,
                                                     scalar_t* result,
                                                     int* result_i) {
  __shared__ scalar_t buf[MAX_SHARED_SCALAR_T];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += THREADS_PER_BLOCK) {
      int end_k = min(m, k2 + THREADS_PER_BLOCK) - k2;
      for (int j = threadIdx.x; j < end_k * 2; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 2 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz[(i * n + j) * 2 + 0];
        scalar_t y1 = xyz[(i * n + j) * 2 + 1];
        int best_i = 0;
        scalar_t best = 1e10;
        int end_ka = end_k & (~2);
        if (end_ka == THREADS_PER_BLOCK) {
          for (int k = 0; k < THREADS_PER_BLOCK; k += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              scalar_t x2 = buf[(k + j) * 2] - x1;
              scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + j;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              scalar_t x2 = buf[(k + j) * 2] - x1;
              scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + j;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2 = buf[k * 2 + 0] - x1;
          scalar_t y2 = buf[k * 2 + 1] - y1;
          scalar_t d = x2 * x2 + y2 * y2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel(
    int b, int n, const scalar_t* xyz1, int m, const scalar_t* xyz2,
    const scalar_t* grad_dist1, const int* idx1, scalar_t* grad_xyz1,
    scalar_t* grad_xyz2) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
      scalar_t x1 = xyz1[(i * n + j) * 2 + 0];
      scalar_t y1 = xyz1[(i * n + j) * 2 + 1];
      int j2 = idx1[i * n + j];
      scalar_t x2 = xyz2[(i * m + j2) * 2 + 0];
      scalar_t y2 = xyz2[(i * m + j2) * 2 + 1];
      scalar_t g = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 1]), -(g * (y1 - y2)));
    }
  }
}

extern "C" diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in,
                     diopiConstTensorHandle_t xyz2_in, diopiTensorHandle_t dist1_out,
                     diopiTensorHandle_t dist2_out, diopiTensorHandle_t idx1_out,
                     diopiTensorHandle_t idx2_out) {
  auto xyz1 = impl::cuda::makeTensor(xyz1_in);
  auto xyz2 = impl::cuda::makeTensor(xyz2_in);
  auto dist1 = impl::cuda::makeTensor(dist1_out);
  auto dist2 = impl::cuda::makeTensor(dist2_out);
  auto idx1 = impl::cuda::makeTensor(idx1_out);
  auto idx2 = impl::cuda::makeTensor(idx2_out);
  int batch_size = xyz1.shape(0);
  int n = xyz1.shape(1);
  int m = xyz2.shape(1);
  // here: wait for dipu ready
  // at::cuda::CUDAGuard device_guard(xyz1.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_dtype(chamfer_distance_forward_cuda_kernel, xyz1_in.dtype(), GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK, stream,
                batch_size, n, xyz1.data(), m,
                xyz2.data(), dist1.data(),
                static_cast<int*>(idx2.data()));
  dispatch_dtype(chamfer_distance_forward_cuda_kernel, xyz1_in.dtype(), GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK, stream,
                batch_size, m, xyz2.data(), n,
                xyz1.data(), dist2.data(),
                static_cast<int*>(idx2.data()));
  return diopiSuccess;
}

extern "C" diopiError_t diopiChamferDistanceBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in,
    diopiConstTensorHandle_t xyz2_in, diopiConstTensorHandle_t idx1_in,
    diopiConstTensorHandle_t idx2_in, diopiConstTensorHandle_t grad_dist1_in,
    diopiConstTensorHandle_t grad_dist2_in, diopiTensorHandle_t grad_xyz1_out,
    diopiTensorHandle_t grad_xyz2_out) {
  auto xyz1 = impl::cuda::makeTensor(xyz1_in);
  auto xyz2 = impl::cuda::makeTensor(xyz2_in);
  auto idx1 = impl::cuda::makeTensor(idx1_in);
  auto idx2 = impl::cuda::makeTensor(idx2_in);
  auto grad_dist1 = impl::cuda::makeTensor(grad_dist1_in);
  auto grad_dist2 = impl::cuda::makeTensor(grad_dist2);
  auto grad_xyz1 = impl::cuda::makeTensor(grad_xyz1_out);
  auto grad_xyz2 = impl::cuda::makeTensor(grad_xyz2_out);
  int batch_size = xyz1.shape(0);
  int n = xyz1.shape(1);
  int m = xyz2.shape(1);
  // here: wait for dipu ready
  // at::cuda::CUDAGuard device_guard(xyz1.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_dtype(chamfer_distance_backward_cuda_kernel, xyz1_in.dtype(), GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK / 2, stream,
                batch_size, m, xyz1.data(), n,
                xyz2.data(), grad_dist1.data(),
                static_cast<int*>(idx1.data()),
                grad_xyz1.data(),
                grad_xyz2.data());
  dispatch_dtype(chamfer_distance_backward_cuda_kernel, xyz1_in.dtype(), GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK / 2, stream,
                batch_size, n, xyz2.data(), m,
                xyz1.data(), grad_dist2.data(),
                static_cast<int*>(idx2.data()),
                grad_xyz2.data(),
                grad_xyz1.data());
  return diopiSuccess;
}
#include <diopi/functions.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include "../helper.hpp"
#include "../cuda_helper.hpp"

#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel_diopi(
    int b, int n, const void* xyz1, int m, const void* xyz2,
    const void* grad_dist1, const int* idx1, void* grad_xyz1,
    void* grad_xyz2) {
  // const scalar_t* xyz1_ = static_cast<const scalar_t*>(xyz1);
  // const scalar_t* xyz2_ = static_cast<const scalar_t*>(xyz2);
  // const scalar_t* grad_dist1_ = static_cast<const scalar_t*>(grad_dist1);
  // scalar_t* grad_xyz1_ = static_cast<scalar_t*>(grad_xyz1);
  // scalar_t* grad_xyz2_ = static_cast<scalar_t*>(grad_xyz2);
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
      const scalar_t* xyz1_ = static_cast<const scalar_t*>(xyz1);
      scalar_t x1 = xyz1_[(i * n + j) * 2 + 0];
      scalar_t y1 = xyz1_[(i * n + j) * 2 + 1];
      int j2 = idx1[i * n + j];
      const scalar_t* xyz2_ = static_cast<const scalar_t*>(xyz2);
      scalar_t x2 = xyz2_[(i * m + j2) * 2 + 0];
      scalar_t y2 = xyz2_[(i * m + j2) * 2 + 1];
      const scalar_t* grad_dist1_ = static_cast<const scalar_t*>(grad_dist1);
      scalar_t g = grad_dist1_[i * n + j] * 2;
      scalar_t* grad_xyz1_ = static_cast<scalar_t*>(grad_xyz1);
      atomicAdd(&(grad_xyz1_[(i * n + j) * 2 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1_[(i * n + j) * 2 + 1]), g * (y1 - y2));
      scalar_t* grad_xyz2_ = static_cast<scalar_t*>(grad_xyz2);
      atomicAdd(&(grad_xyz2_[(i * m + j2) * 2 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2_[(i * m + j2) * 2 + 1]), -(g * (y1 - y2)));
    }
  }
}

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel_diopi2(
    int b, int n, const void* xyz1_, int m, const void* xyz2_,
    const void* grad_dist1_, const int* idx1, void* grad_xyz1_,
    void* grad_xyz2_) {
  const scalar_t* xyz1 = static_cast<const scalar_t*>(xyz1_);
  const scalar_t* xyz2 = static_cast<const scalar_t*>(xyz2_);
  const scalar_t* grad_dist1 = static_cast<const scalar_t*>(grad_dist1_);
  scalar_t* grad_xyz1 = static_cast<scalar_t*>(grad_xyz1_);
  scalar_t* grad_xyz2 = static_cast<scalar_t*>(grad_xyz2_);

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

// DIOPI_API diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in, diopiConstTensorHandle_t xyz2_in, diopiTensorHandle_t dist1_out,
//                                             diopiTensorHandle_t dist2_out, diopiTensorHandle_t idx1_out, diopiTensorHandle_t idx2_out);

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
  int batch_size = xyz1.size(0);
  std::cout << "dkx fwd batch_size" << batch_size << std::endl;
  int n = xyz1.size(1);
  int m = xyz2.size(1);
  std::cout << "dkx fwd n" << n << std::endl;
  std::cout << "dkx fwd m" << m << std::endl;
  // here: wait for dipu ready
  // // at::cuda::CUDAGuard device_guard(xyz1.device());
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(chamfer_distance_forward_cuda_kernel_diopi, xyz1.dtype(), GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK, stream,
                batch_size, n, xyz1.data(), m,
                xyz2.data(), dist1.data(),
                static_cast<int*>(idx1.data()));
  dispatch_float_types_and_half(chamfer_distance_forward_cuda_kernel_diopi, xyz1.dtype(), GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK, stream,
                batch_size, m, xyz2.data(), n,
                xyz1.data(), dist2.data(),
                static_cast<int*>(idx2.data()));
  return diopiSuccess;
}

// extern "C" {
//     c10::DeviceType device2DeviceType(const diopiDevice_t device);
// }

// DIOPI_API diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1, diopiConstTensorHandle_t xyz2,
//                                             diopiConstTensorHandle_t idx1, diopiConstTensorHandle_t idx2, diopiConstTensorHandle_t grad_dist1, diopiConstTensorHandle_t grad_dist2,
//                                             diopiTensorHandle_t grad_xyz1, diopiTensorHandle_t grad_xyz2);

// extern c 和声明不一致。 DIOPI_API 这是一个 __attribute__((weak)) 的声明。
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
  auto grad_dist2 = impl::cuda::makeTensor(grad_dist2_in);
  auto grad_xyz1 = impl::cuda::makeTensor(grad_xyz1_out);
  auto grad_xyz2 = impl::cuda::makeTensor(grad_xyz2_out);
  int batch_size = xyz1.size(0);
  std::cout << "dkx bwd batch_size" << batch_size << std::endl;
  int n = xyz1.size(1);
  int m = xyz2.size(1);
  std::cout << "dkx bwd n" << n << std::endl;
  std::cout << "dkx bwd m" << m << std::endl;
  // here: wait for dipu ready
  //// at::cuda::CUDAGuard device_guard(device2DeviceType(xyz1.device()));
  auto stream = impl::cuda::getStream(ctx);
  // dispatch_float_types_and_half(
  //               chamfer_distance_backward_cuda_kernel_diopi,
  //               xyz1.dtype(),
  //               GET_BLOCKS(batch_size * n),
  //               THREADS_PER_BLOCK / 2,
  //               stream,
  //               batch_size, m, xyz1.data(), n,
  //               xyz2.data(), grad_dist1.data(),
  //               static_cast<const int*>(idx1.data()),
  //               grad_xyz1.data(),
  //               grad_xyz2.data());
  dispatch_float_types_and_half(
                chamfer_distance_backward_cuda_kernel_diopi2,
                xyz1.dtype(),
                GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK / 2, stream,
                batch_size, m, xyz1.data(), n,
                xyz2.data(), grad_dist1.data(),
                static_cast<const int*>(idx1.data()), grad_xyz1.data(),
                grad_xyz2.data());
  // dispatch_float_types_and_half(chamfer_distance_backward_cuda_kernel_diopi,
  //               xyz1.dtype(),
  //               GET_BLOCKS(batch_size * m),
  //               THREADS_PER_BLOCK / 2,
  //               stream,
  //               batch_size, n, xyz2.data(), m,
  //               xyz1.data(), grad_dist2.data(),
  //               static_cast<const int*>(idx2.data()),
  //               grad_xyz2.data(),
  //               grad_xyz1.data());
  dispatch_float_types_and_half(chamfer_distance_backward_cuda_kernel_diopi2,
                xyz1.dtype(),
                GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK / 2, stream,
                batch_size, n, xyz2.data(), m,
                xyz1.data(), grad_dist2.data(),
                static_cast<const int*>(idx2.data()), grad_xyz2.data(),
                grad_xyz1.data());
  return diopiSuccess;
}
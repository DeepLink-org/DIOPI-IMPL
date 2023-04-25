#include <cuda_runtime.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include <stdio.h>
#include<assert.h>

#include <iostream>
#include <vector>

#include "../cuda_helper.hpp"
#include "../helper.hpp"

namespace impl {

namespace cuda {

template <typename scalar_t>
__global__ void ball_query_forward_cuda_kernel_diopi(int b, int n, int m,
                                                    float min_radius,
                                                    float max_radius, int nsample,
                                                    const void* new_xyz_, const void* xyz_,
                                                    int* idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)

  const scalar_t* new_xyz=static_cast<const scalar_t*>(new_xyz_);
  const scalar_t* xyz=static_cast<const scalar_t*>(xyz_);

  int bs_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, m) {
    if (bs_idx >= b) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float max_radius2 = max_radius * max_radius;
    float min_radius2 = min_radius * min_radius;
    scalar_t new_x = new_xyz[0];
    scalar_t new_y = new_xyz[1];
    scalar_t new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
      scalar_t x = xyz[k * 3 + 0];
      scalar_t y = xyz[k * 3 + 1];
      scalar_t z = xyz[k * 3 + 2];
      scalar_t d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
             (new_z - z) * (new_z - z);
      if (d2 == 0 || (d2 >= min_radius2 && d2 < max_radius2)) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[l] = k;
          }
        }
        idx[cnt] = k;
        ++cnt;
        if (cnt >= nsample) break;
      }
    }
  }
}

}  // namespace cuda

}  // namespace impl

DIOPI_API diopiError_t diopiBallQuery(diopiContextHandle_t ctx, diopiTensorHandle_t idx_,
                        diopiConstTensorHandle_t new_xyz_, diopiConstTensorHandle_t xyz_, 
                        int64_t b, int64_t n, int64_t m, int64_t nsample,
                        float min_radius, float max_radius) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  auto new_xyz = impl::cuda::makeTensor(new_xyz_);
  auto xyz = impl::cuda::makeTensor(xyz_);
  auto idx = impl::cuda::makeTensor(idx_);

//   at::cuda::CUDAGuard device_guard(new_xyz.device());
  auto stream = impl::cuda::getStream(ctx);

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(m, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);


  dispatch_float_types_and_half(impl::cuda::ball_query_forward_cuda_kernel_diopi,
                                new_xyz.dtype(),
                                blocks,
                                threads,
                                stream,
                                b, 
                                n,
                                m,
                                min_radius, 
                                max_radius, 
                                nsample,
                                new_xyz.data(), 
                                xyz.data(),
                                (int*)(idx.data()));
  return diopiSuccess;
}

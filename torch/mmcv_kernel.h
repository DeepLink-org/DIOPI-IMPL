/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_TORCH_FUNCTIONS_MMCV_H_
#define IMPL_TORCH_FUNCTIONS_MMCV_H_

#include <ATen/ATen.h>

namespace mmcv {
namespace ops {

using namespace at;

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset);

}  // namespace ops
}  // namespace mmcv


#endif  // IMPL_TORCH_FUNCTIONS_MMCV_H_
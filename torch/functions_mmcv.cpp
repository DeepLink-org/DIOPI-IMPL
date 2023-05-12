/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <iostream>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "context.h"
#include "helper.hpp"
#include "mmcv_kernel.h"


extern "C" {

diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out,
                      diopiConstTensorHandle_t dets,
                      diopiConstTensorHandle_t scores, double iouThreshold,
                      int64_t offset) {
  auto atDets = impl::aten::buildATen(dets);
  auto atScores = impl::aten::buildATen(scores);
  auto atOut = mmcv::ops::NMSCUDAKernelLauncher(atDets, atScores, iouThreshold, offset);
  impl::aten::buildDiopiTensor(ctx, atOut, out);
}

diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output_, diopiTensorHandle_t argmax_y_,
                               diopiTensorHandle_t argmax_x_, diopiConstTensorHandle_t input_, diopiConstTensorHandle_t rois_,
                               int64_t aligned_height, int64_t aligned_width, int64_t sampling_ratio, int64_t pool_mode,
                               float spatial_scale, bool aligned) {
  auto input = impl::aten::buildATen(input_);
  auto rois = impl::aten::buildATen(rois_);
  auto output = impl::aten::buildATen(output_);
  auto argmax_y = impl::aten::buildATen(argmax_y_);
  auto argmax_x = impl::aten::buildATen(argmax_x_);
  mmcv::ops::ROIAlignForwardCUDAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned);
}

diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input_, diopiConstTensorHandle_t grad_output_,
                                       diopiConstTensorHandle_t rois_, diopiConstTensorHandle_t argmax_y_, diopiConstTensorHandle_t argmax_x_,
                                       int64_t aligned_height, int64_t aligned_width, int64_t sampling_ratio, int64_t pool_mode, float spatial_scale,
                                       bool aligned) {
  auto grad_output = impl::aten::buildATen(grad_output_);
  auto rois = impl::aten::buildATen(rois_);
  auto argmax_y = impl::aten::buildATen(argmax_y_);
  auto argmax_x = impl::aten::buildATen(argmax_x_);
  auto grad_input = impl::aten::buildATen(grad_input_);
  mmcv::ops::ROIAlignBackwardCUDAKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}

}  // extern "C"
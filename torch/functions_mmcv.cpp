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

}  // extern "C"
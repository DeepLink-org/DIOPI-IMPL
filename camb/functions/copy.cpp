/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {
    // TODO(waiting for dispatch): support broadcast, dealing with uncontiguous
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tr = impl::camb::DiopiTensor(input);
    auto src_tr = impl::camb::DiopiTensor(src);

    if (src_tr.dtype() != input_tr.dtype()) {
        dataTypeCast(ctx, src_tr, input_tr.dtype());
    }

    CnnlTensorDesc input_desc(input_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc src_desc(src_tr, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECKCNNL(cnnlCopy(handle, src_desc.get(), src_tr.data(), input_desc.get(), input_tr.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

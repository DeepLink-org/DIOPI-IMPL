#include "../cnnl_helper.hpp"
#include "diopi/functions.h"

DIOPI_API diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto trOut = impl::camb::makeTensor(out);
    auto trInput = impl::camb::makeTensor(input);
    std::vector<int32_t> targetShape(size.data, size.data + size.len);

    // check whether input.shape() match the targetShape
    const std::vector<int32_t> inputShape = trInput.shape();
    int32_t nDimsTarget = targetShape.size();
    int32_t nDimsInput = inputShape.size();
    bool flag = false;
    for (int32_t i = 0; i < nDimsTarget; ++i) {
        int32_t idxT = nDimsTarget - i - 1;
        int32_t idxI = nDimsInput - i - 1;
        if (flag) {
            DIOPI_CHECK(inputShape[idxI] == 1, "shape1 not match shape2, can't broadcast");
        }
        if (idxI < 0) {
            break;
        }
        if (targetShape[idxT] == inputShape[idxI]) {
            continue;
        }
        if (inputShape[idxI] == 1) {
            flag = 1;
            continue;
        }
        DIOPI_CHECK(false, "shape1 not match shape2, can't broadcast");
    }

    CnnlTensorDesc inputDesc(trInput, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(trOut, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlExpand(handle, inputDesc.get(), trInput.data(), outDesc.get(), trOut.data()));
    return diopiSuccess;
}
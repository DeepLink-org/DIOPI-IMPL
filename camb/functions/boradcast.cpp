#include "../cnnl_helper.hpp"
#include "diopi/functions.h"
namespace impl {
namespace camb {

using DiopiTensorT = DiopiTensor<diopiTensorHandle_t>;

diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensorT& out, const DiopiTensorT& input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    // check whether input.shape() match the targetShape
    std::vector<int32_t> targetShape = out.shape();
    const std::vector<int32_t> inputShape = input.shape();
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

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlExpand(handle, inputDesc.get(), const_cast<DiopiTensorT&>(input).data(), outDesc.get(), out.data()));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

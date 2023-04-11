/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"

namespace impl {
namespace camb {

extern "C" DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t w,
                                           diopiTensorHandle_t dw,
                                           diopiTensorHandle_t buf,
                                           double lr,
                                           double momentum,
                                           double dampening,
                                           double weight_decay,
                                           bool nesterov) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor w_tensor(w);
    DiopiTensor dw_tensor(dw);
    DiopiTensor buf_tensor(buf);

    std::vector<DiopiTensor *> pTensors{&dw_tensor, &buf_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor w_tensor_tmp = w_tensor;
    if (dw_tensor.dtype() != w_tensor_tmp.dtype()) {
        w_tensor_tmp = requiresTensor(ctx, w_tensor.shape(), dw_tensor.dtype());
    }

    CnnlTensorDesc w_desc_tmp(w_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc dw_desc(dw_tensor, CNNL_LAYOUT_ARRAY);

    // a = a * scale_a + b * scale_b;
    auto add_mul_func = [&](auto &a, float scale_a, auto b, float scale_b) {
        size_t workspace_size;
        std::vector<int> shape;
        shape.push_back(a.numel());
        CnnlTensorDesc a_desc, b_desc;
        DIOPI_CALL(a_desc.set(a, CNNL_LAYOUT_ARRAY, shape));
        DIOPI_CALL(b_desc.set(b, CNNL_LAYOUT_ARRAY, shape));

        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, b_desc.get(), a_desc.get(), &workspace_size));

        void *workspace = nullptr;
        if (workspace_size != 0) {
            workspace = requiresBuffer(ctx, workspace_size).data();
        }

        DIOPI_CALLCNNL(cnnlBiasAdd(handle, &scale_b, b_desc.get(), b.data(), workspace, workspace_size, &scale_a, a_desc.get(), a.data()));
        return diopiSuccess;
    };

    if (weight_decay != 0) {
        DIOPI_CALL(add_mul_func(dw_tensor, 1.0, w_tensor_tmp, weight_decay));
    }
    if (momentum != 0) {
        if (buf == nullptr) {
            if (nesterov) {
                DIOPI_CALL(add_mul_func(dw_tensor, 1.0, dw_tensor, momentum));
            }
        } else {
            CnnlTensorDesc buf_desc(buf_tensor, CNNL_LAYOUT_ARRAY);
            DIOPI_CALL(add_mul_func(buf_tensor, momentum, dw_tensor, (1.0 - dampening)));
            if (nesterov) {
                DIOPI_CALL(add_mul_func(dw_tensor, 1.0, buf_tensor, momentum));
            } else {
                DIOPI_CALLCNNL(cnnlCopy(handle, buf_desc.get(), buf_tensor.data(), dw_desc.get(), dw_tensor.data()));
            }
        }
    }

    std::vector<int64_t> shape{1};
    diopiSize_t size(shape.data(), shape.size());
    DiopiTensor lr_tensor;
    diopiScalar_t lr_scalar{diopi_dtype_float64, {lr}};
    makeTensorFromScalar(ctx, &lr_scalar, lr_tensor);
    dataTypeCast(ctx, lr_tensor, dw_tensor.dtype());
    DIOPI_CALLCNNL(cnnlGradientDescent(handle, dw_desc.get(), dw_tensor.data(), lr_tensor.data(), w_desc_tmp.get(), w_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, w_tensor, w_tensor_tmp));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

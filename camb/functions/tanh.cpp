#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "activation_internal.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhBackward(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t grad_input,
                                          diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_output_tensor = DiopiTensor(grad_output);
    auto output_tensor = DiopiTensor(output);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    cnnl_activation_backward_internal(ctx, grad_input_tensor, grad_output_tensor, output_tensor, attr);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

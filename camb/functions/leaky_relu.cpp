#include <diopi/functions.h>

#include <iostream>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {
diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    if (input_tensor.dtype() == diopi_dtype_float64) {
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc x_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc y_desc(output_tensor, CNNL_LAYOUT_ARRAY);
    void* x_ptr = input_tensor.data();
    void* y_ptr = output_tensor.data();

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> CnnlActivation;
    cnnlActivationDescriptor_t activation_desc = CnnlActivation.get();
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v5(
        activation_desc, CNNL_ACTIVATION_LEAKYRELU, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, negative_slope->fval, 0, 0.0, 0.0, true));
    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc, NULL, x_desc.get(), x_ptr, NULL, y_desc.get(), y_ptr));
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    diopiLeakyRelu(ctx, input, input, negative_slope);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx,
                                    diopiTensorHandle_t grad_input,
                                    diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t output,
                                    const diopiScalar_t* negative_slope,
                                    bool input_is_result) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_grad = DiopiTensor(grad_input);
    auto output_grad = DiopiTensor(grad_output);
    auto output_tensor = DiopiTensor(output);

    CnnlTensorDesc input_grad_desc(input_grad, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_grad_desc(output_grad, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);

    void* output_ptr = output_tensor.data();
    void* output_grad_ptr = output_grad.data();
    void* input_grad_ptr = input_grad.data();

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> CnnlActivation;
    cnnlActivationDescriptor_t activation_desc = CnnlActivation.get();
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v5(activation_desc,
                                                  CNNL_ACTIVATION_LEAKYRELU,
                                                  CNNL_ACTIVATION_HIGH_PRECISION,
                                                  CNNL_NOT_PROPAGATE_NAN,
                                                  negative_slope->fval,
                                                  0,
                                                  0.0,
                                                  0.0,
                                                  input_is_result));
    DIOPI_CALLCNNL(cnnlActivationBackward(handle,
                                          activation_desc,
                                          NULL,
                                          output_desc.get(),
                                          NULL,
                                          output_grad_desc.get(),
                                          output_grad_ptr,
                                          input_grad_desc.get(),
                                          input_grad_ptr,
                                          NULL,
                                          input_grad_desc.get(),
                                          input_grad_ptr));
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl

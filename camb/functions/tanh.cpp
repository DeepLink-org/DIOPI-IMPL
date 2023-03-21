#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activate_desc;

    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(activate_desc.get(), CNNL_ACTIVATION_TANH, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));

    if (input_tensor.dtype() == diopi_dtype_float64) {
        diopiTensorHandle_t input_ = const_cast<diopiTensorHandle_t>(input);
        auto input_tensor_f32 = dataTypeCast(ctx, input_tensor, diopi_dtype_float32);
        CnnlTensorDesc f32_desc(input_tensor_f32, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlActivationForward(
            handle, activate_desc.get(), nullptr, f32_desc.get(), input_tensor_f32.data(), nullptr, f32_desc.get(), input_tensor_f32.data()));
        dataTypeCast(ctx, output_tensor, input_tensor_f32);
    } else {
        CnnlTensorDesc in_desc(input_tensor, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc out_desc(output_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(
            cnnlActivationForward(handle, activate_desc.get(), nullptr, in_desc.get(), input_tensor.data(), nullptr, out_desc.get(), output_tensor.data()));
    }

    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activate_desc;

    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(activate_desc.get(), CNNL_ACTIVATION_TANH, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));

    if (input_tensor.dtype() == diopi_dtype_float64) {
        diopiTensorHandle_t input_ = const_cast<diopiTensorHandle_t>(input);
        auto input_tensor_f32 = dataTypeCast(ctx, input_tensor, diopi_dtype_float32);
        CnnlTensorDesc f32_desc(input_tensor_f32, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlActivationForward(
            handle, activate_desc.get(), nullptr, f32_desc.get(), input_tensor_f32.data(), nullptr, f32_desc.get(), input_tensor_f32.data()));
        dataTypeCast(ctx, input_tensor, input_tensor_f32);
    } else {
        CnnlTensorDesc in_desc(input_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(
            cnnlActivationForward(handle, activate_desc.get(), nullptr, in_desc.get(), input_tensor.data(), nullptr, in_desc.get(), input_tensor.data()));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhBackward(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t grad_input,
                                          diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_grad = DiopiTensor(grad_input);
    auto output_grad = DiopiTensor(grad_output);
    auto output_tensor = DiopiTensor(output);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> CnnlActivation;
    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(CnnlActivation.get(), CNNL_ACTIVATION_TANH, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));

    if (output_grad.dtype() == diopi_dtype_float64) {
        auto input_grad_f32 = dataTypeCast(ctx, input_grad, diopi_dtype_float32);
        auto output_grad_f32 = dataTypeCast(ctx, output_grad, diopi_dtype_float32);
        auto output_tensor_f32 = dataTypeCast(ctx, output_tensor, diopi_dtype_float32);

        CnnlTensorDesc input_grad_desc(input_grad_f32, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc output_grad_desc(output_grad_f32, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc output_desc(output_tensor_f32, CNNL_LAYOUT_ARRAY);

        DIOPI_CALLCNNL(cnnlActivationBackward(handle,
                                              CnnlActivation.get(),
                                              nullptr,
                                              output_desc.get(),
                                              output_tensor_f32.data(),
                                              output_grad_desc.get(),
                                              output_grad_f32.data(),
                                              input_grad_desc.get(),
                                              input_grad_f32.data(),
                                              nullptr,
                                              input_grad_desc.get(),
                                              input_grad_f32.data()));

        dataTypeCast(ctx, output_tensor, input_grad_f32);
    } else {
        CnnlTensorDesc input_grad_desc(input_grad, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc output_grad_desc(output_grad, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);

        DIOPI_CALLCNNL(cnnlActivationBackward(handle,
                                              CnnlActivation.get(),
                                              nullptr,
                                              output_desc.get(),
                                              output_tensor.data(),
                                              output_grad_desc.get(),
                                              output_grad.data(),
                                              input_grad_desc.get(),
                                              input_grad.data(),
                                              nullptr,
                                              input_grad_desc.get(),
                                              input_grad.data()));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

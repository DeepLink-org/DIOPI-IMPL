#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {
extern "C" {

diopiError_t clampCommon(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);
    DIOPI_CHECK(input_tensor.dtype() == output_tensor.dtype(), "the dtype of input and output must be the same")

    auto input32_tensor = input_tensor;
    auto output32_tensor = output_tensor;
    if (input_tensor.dtype() == diopi_dtype_int64 || input_tensor.dtype() == diopi_dtype_int16 || input_tensor.dtype() == diopi_dtype_int8) {
        input32_tensor = dataTypeCast(ctx, input_tensor, diopi_dtype_int32);
        output32_tensor = dataTypeCast(ctx, output_tensor, diopi_dtype_int32);
    }
    CnnlTensorDesc input32Desc(input32_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output32Desc(output32_tensor, CNNL_LAYOUT_ARRAY);

    void * min_ptr = nullptr;
    void * max_ptr = nullptr;
    if (nullptr != min) {
        auto min_tensor = DiopiTensor(min);
        if (input32_tensor.dtype() != min_tensor.dtype()) {
            min_tensor = dataTypeCast(ctx, min_tensor, input32_tensor.dtype());
        }
        min_ptr = min_tensor.data();
    }
    if (nullptr != max) {
        auto max_tensor = DiopiTensor(max);
        if (input32_tensor.dtype() != max_tensor.dtype()) {
            max_tensor = dataTypeCast(ctx, max_tensor, input32_tensor.dtype());
        }
        max_ptr = max_tensor.data();
    }

    DIOPI_CALLCNNL(cnnlClip_v2(handle, CNNL_POINTER_MODE_DEVICE, input32Desc.get(), input32_tensor.data(), min_ptr, max_ptr, output32Desc.get(), output32_tensor.data()));
    if (output_tensor.dtype() != output32_tensor.dtype()) {
        dataTypeCast(ctx, output_tensor, output32_tensor);
    }
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiTensorHandle_t min_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, min));
    diopiTensorHandle_t max_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, max));
    return clampCommon(ctx, input, input, min_tensor, max_tensor);
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, input, min, max);
}

diopiError_t diopiClampScalar(
    diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiTensorHandle_t min_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, min));
    diopiTensorHandle_t max_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, max));
    return clampCommon(ctx, input, out, min_tensor, max_tensor);
}

diopiError_t diopiClamp(
    diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, out, min, max);
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    diopiTensorHandle_t max_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, max));
    return clampCommon(ctx, input, input, nullptr, max_tensor);
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, input, nullptr, max);
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    diopiTensorHandle_t max_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, max));
    return clampCommon(ctx, input, out, nullptr, max_tensor);
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, out, nullptr, max);
}

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    diopiTensorHandle_t min_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, min));
    return clampCommon(ctx, input, input, min_tensor, nullptr);
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    return clampCommon(ctx, input, input, min, nullptr);
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    diopiTensorHandle_t min_tensor = diopiTensorHandle_t(makeTensorFromScalar(ctx, min));
    return clampCommon(ctx, input, out, min_tensor, nullptr);
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    return clampCommon(ctx, input, out, min, nullptr);
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

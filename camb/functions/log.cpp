#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t LogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, cnnlLogBase_t log_base) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        diopiTensorHandle_t out_cast;
        diopiSize_t input_shape;
        DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
        DIOPI_CALL(diopiRequireTensor(ctx, &out_cast, &input_shape, nullptr, diopi_dtype_float32, diopi_device));
        auto out_cast_tensor = makeTensor(out_cast);
        CnnlTensorDesc out_cast_desc(out_cast_tensor, CNNL_LAYOUT_ARRAY);

        auto input_cast_tensor = dataTypeCast(ctx, input_tensor, diopi_dtype_float32);
        CnnlTensorDesc input_cast_desc(input_cast_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlLog(handle, log_base, input_cast_desc.get(), input_cast_tensor.data(), out_cast_desc.get(), out_cast_tensor.data()));
        DIOPI_CHECKCNNL(
            cnnlCastDataType(handle, out_cast_desc.get(), out_cast_tensor.data(), CNNL_CAST_FLOAT_TO_DOUBLE, input_desc.get(), input_tensor.data()));
    } else {
        DIOPI_CALLCNNL(cnnlLog(handle, log_base, input_desc.get(), input_tensor.data(), input_desc.get(), input_tensor.data()));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t Log(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, cnnlLogBase_t log_base) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto out_tensor = makeTensor(out);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        diopiTensorHandle_t out_cast;
        diopiSize_t input_shape;
        DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
        DIOPI_CALL(diopiRequireTensor(ctx, &out_cast, &input_shape, nullptr, diopi_dtype_float32, diopi_device));
        auto out_cast_tensor = makeTensor(out_cast);
        CnnlTensorDesc out_cast_desc(out_cast_tensor, CNNL_LAYOUT_ARRAY);

        diopiTensorHandle_t input_t = const_cast<diopiTensorHandle_t>(input);
        auto input_t_tensor = makeTensor(input_t);
        CnnlTensorDesc input_t_desc(input_t_tensor, CNNL_LAYOUT_ARRAY);
        auto input_cast_tensor = dataTypeCast(ctx, input_t_tensor, diopi_dtype_float32);
        CnnlTensorDesc input_cast_desc(input_cast_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlLog(handle, log_base, input_cast_desc.get(), input_cast_tensor.data(), out_cast_desc.get(), out_cast_tensor.data()));
        DIOPI_CHECKCNNL(cnnlCastDataType(handle, out_cast_desc.get(), out_cast_tensor.data(), CNNL_CAST_FLOAT_TO_DOUBLE, out_desc.get(), out_tensor.data()));
    } else {
        DIOPI_CALLCNNL(cnnlLog(handle, log_base, input_desc.get(), input_tensor.data(), out_desc.get(), out_tensor.data()));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { DIOPI_CALL(LogInp(ctx, input, CNNL_LOG_E)); }

DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(Log(ctx, out, input, CNNL_LOG_E));
}

DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { DIOPI_CALL(LogInp(ctx, input, CNNL_LOG_2)); }

DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(Log(ctx, out, input, CNNL_LOG_2));
}

DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { DIOPI_CALL(LogInp(ctx, input, CNNL_LOG_10)); }

DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(Log(ctx, out, input, CNNL_LOG_10));
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
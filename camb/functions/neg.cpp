
#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        auto input_cast_tensor = dataTypeCast(ctx, input_tensor, diopi_dtype_float32);
        CnnlTensorDesc input_cast_desc(input_cast_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlNegTensor(handle, input_cast_desc.get(), input_cast_tensor.data(), input_cast_desc.get(), input_cast_tensor.data()));
        dataTypeCast(ctx, input_tensor, input_cast_tensor);
    } else {
        CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlNegTensor(handle, input_desc.get(), input_tensor.data(), input_desc.get(), input_tensor.data()));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);

    if (input_tensor.dtype() == diopi_dtype_float64) {
        diopiTensorHandle_t input_t = const_cast<diopiTensorHandle_t>(input);
        auto input_t_tensor = DiopiTensor(input_t);
        CnnlTensorDesc input_t_desc(input_t_tensor, CNNL_LAYOUT_ARRAY);
        auto input_cast_tensor = dataTypeCast(ctx, input_t_tensor, diopi_dtype_float32);
        CnnlTensorDesc input_cast_desc(input_cast_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlNegTensor(handle, input_cast_desc.get(), input_cast_tensor.data(), input_cast_desc.get(), input_cast_tensor.data()));
        dataTypeCast(ctx, out_tensor, input_cast_tensor);
    } else {
        CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlNegTensor(handle, input_desc.get(), input_tensor.data(), out_desc.get(), out_tensor.data()));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

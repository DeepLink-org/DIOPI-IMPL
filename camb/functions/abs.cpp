#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        set_last_error_string("%s", "cnnlAbs function not support float64.");
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlAbs(handle, input_desc.get(), input_tensor.data(), input_desc.get(), input_tensor.data()));
}

extern "C" diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto output_tensor = makeTensor(out);
    if (output_tensor.dtype() == diopi_dtype_float64) {
        set_last_error_string("%s", "cnnlAbs function not support float64.");
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlAbs(handle, input_desc.get(), input_tensor.data(), output_desc.get(), output_tensor.data()));
}

}  // namespace camb
}  // namespace impl

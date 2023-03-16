#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    auto handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        set_last_error_string("%s", "cnnlAbs function not support float64.");
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, input_desc.get(), input_tensor.data(), input_desc.get(), input_tensor.data()));
    return diopiSuccess;
}

extern "C" diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    auto handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto output_tensor = makeTensor(out);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        set_last_error_string("%s", "cnnlAbs function not support float64.");
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, input_desc.get(), input_tensor.data(), output_desc.get(), output_tensor.data()));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

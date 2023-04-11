/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {
extern "C" {
DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    std::vector<DiopiTensor*> pinput{&input_tensor};
    std::set<diopiDtype_t> supported_dtypes{diopi_dtype_bool,
                                            diopi_dtype_uint8,
                                            diopi_dtype_int8,
                                            diopi_dtype_uint16,
                                            diopi_dtype_int16,
                                            diopi_dtype_uint32,
                                            diopi_dtype_int32,
                                            diopi_dtype_uint64,
                                            diopi_dtype_int64,
                                            diopi_dtype_float16,
                                            diopi_dtype_float32,
                                            diopi_dtype_float64};

    DIOPI_CALL(autoCastTensorType(ctx, pinput, supported_dtypes));
    DiopiTensor out_tensor_temp = out_tensor;
    if (out_tensor_temp.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_temp, input_tensor.dtype()));
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(
        cnnlUnfold(handle, int32_t(dim), int32_t(size), int32_t(step), input_desc.get(), input_tensor.data(), out_desc.get(), out_tensor_temp.data()));
    if (out_tensor_temp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_temp));
    }
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl

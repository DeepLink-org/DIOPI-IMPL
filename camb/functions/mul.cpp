#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {
DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, 
                                diopiTensorHandle_t out, 
                                diopiConstTensorHandle_t input, 
                                diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto other_tensor = makeTensor(other);
    auto out_tensor = makeTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    // broadcast todo
    cnnlTensorDescriptor_t input_descs[] = {input_desc.get(), other_desc.get()};
    const void* inputs[] = {input_tensor.data(), other_tensor.data()};
    DIOPI_CALLCNNL(cnnlMulN(handle, input_descs, inputs, 2, out_desc.get(), out_tensor.data()))
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, 
                                   diopiTensorHandle_t input, 
                                   diopiConstTensorHandle_t other) {
    diopiMul(ctx, input, input, other);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, 
                                      diopiTensorHandle_t out, 
                                      diopiConstTensorHandle_t input, 
                                      const diopiScalar_t* other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto out_tensor = makeTensor(out);

    diopiTensorHandle_t other_t;
    diopiSize_t input_shape;
    DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
    DIOPI_CALL(diopiRequireTensor(ctx, &other_t, &input_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, other_t, other));
    auto other_t_tensor = makeTensor(other_t);
    CnnlTensorDesc other_t_desc(other_t_tensor, CNNL_LAYOUT_ARRAY);

    diopiMul(ctx, out, input, other_t);
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, 
                                         diopiTensorHandle_t input, 
                                         const diopiScalar_t* other) {
    diopiMulScalar(ctx, input, input, other);
}

}  // extern "C"
}  // namespace camb
}  // namespace impl

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

using DiopiTensorT = DiopiTensor<diopiTensorHandle_t>;
diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensorT& out, const DiopiTensorT& input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // check whether input.shape() match the targetShape
    std::vector<int64_t> targetShape = out.shape();
    std::vector<int64_t> inputShape = input.shape();
    int64_t nDimsTarget = targetShape.size();
    int64_t nDimsInput = inputShape.size();
    if (nDimsInput < nDimsTarget) {
        inputShape.insert(inputShape.begin(), nDimsTarget - nDimsInput, 1);
    }

    for (int i = 0; i < nDimsTarget; i++) {
        DIOPI_CHECK( ((inputShape[i] == 1) || (inputShape[i] == targetShape[i])), "shape1 not match shape2, can't broadcast");
    }
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlExpand(handle, inputDesc.get(), const_cast<DiopiTensorT&>(input).data(), outDesc.get(), out.data()));
    return diopiSuccess;
}

DiopiTensorT broadcastHelper(diopiContextHandle_t ctx, DiopiTensorT input_tensor, DiopiTensorT target_tensor) {
    diopiTensorHandle_t bcast_input = nullptr;
    DiopiTensorT bcast_input_tensor;
    if (input_tensor.shape() != target_tensor.shape()) {
        bcast_input_tensor = requiresTensor(ctx, vec2diopiSize_t(target_tensor.shape()), target_tensor.dtype());
        broadcast(ctx, bcast_input_tensor, input_tensor);
        return bcast_input_tensor;
    } else {
        bcast_input_tensor = input_tensor;
        return bcast_input_tensor;
    }
}

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    diopiTensorHandle_t other_ = diopiTensorHandle_t(other);

    auto input_tensor = makeTensor(input_);
    auto other_tensor = makeTensor(other_);
    auto out_tensor = makeTensor(out);
    DiopiTensor<diopiTensorHandle_t> out_tensor_tmp;
    if ((out_tensor.dtype() != diopi_dtype_float16) && (out_tensor.dtype() != diopi_dtype_float32)) {
        out_tensor_tmp = dataTypeCast(ctx, out_tensor, diopi_dtype_float16);
    } else {
        out_tensor_tmp = makeTensor(out);
    }
    input_tensor = dataTypeCast(ctx, input_tensor, out_tensor_tmp.dtype());
    other_tensor = dataTypeCast(ctx, other_tempensor, out_tensor_tmp.dtype());

    DiopiTensorT bcast_input_tensor = broadcastHelper(ctx, input_tensor, out_tensor_tmp);
    DiopiTensorT bcast_other_tempensor = broadcastHelper(ctx, other_tempensor, out_tensor_tmp);

    CnnlTensorDesc bcast_input_desc(bcast_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc bcast_other_desc(bcast_other_tempensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    cnnlTensorDescriptor_t input_descs[] = {bcast_input_desc.get(), bcast_other_desc.get()};
    const void* inputs[] = {bcast_input_tensor.data(), bcast_other_tempensor.data()};

    DIOPI_CALLCNNL(cnnlMulN(handle, input_descs, inputs, 2, out_desc.get(), out_tensor_tmp.data()))
    if (out_tensor_tmp.dtype() != out_tensor.dtype()) {
        dataTypeCast(ctx, out_tensor, out_tensor_tmp);
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiMul(ctx, input, input, other);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto out_tensor = makeTensor(out);

    diopiTensorHandle_t other_temp = requiresTensor(ctx, vec2diopiSize_t(input_tensor.shape()), input_tensor.dtype());
    DIOPI_CALL(diopiFill(ctx, other_temp, other));
    auto other_temp_tensor = makeTensor(other_temp);
    CnnlTensorDesc other_temp_desc(other_temp_tensor, CNNL_LAYOUT_ARRAY);
    diopiMul(ctx, out, input, other_temp);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiMulScalar(ctx, input, input, other);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl

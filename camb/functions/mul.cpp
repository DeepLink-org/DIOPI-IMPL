#include <diopi/functions.h>
#include "../common/common.hpp"
#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

using DiopiTensorT = DiopiTensor<diopiTensorHandle_t>;
diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensorT& out, const DiopiTensorT& input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // check whether input.shape() match the targetShape
    std::vector<int32_t> targetShape(out.shape().begin(), out.shape().end());
    std::vector<int32_t> inputShape(input.shape().begin(), input.shape().end());

    int32_t nDimsTarget = targetShape.size();
    int32_t nDimsInput = inputShape.size();
    if (nDimsInput < nDimsTarget) {
        inputShape.insert(inputShape.begin(), nDimsTarget - nDimsInput, 1);
    }

    for (int i = 0; i < nDimsTarget; i++) {
        if ((inputShape[i] != 1) && (inputShape[i] != targetShape[i])) {
            DIOPI_CHECK(false, "shape1 not match shape2, can't broadcast");
            break;
        }
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
        diopiSize_t target_size;
        diopiGetTensorShape(target_tensor, &target_size);
        diopiRequireTensor(ctx, &bcast_input, &target_size, nullptr, target_tensor.dtype(), diopi_device);
        bcast_input_tensor = makeTensor(bcast_input);
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
    }
    else {
        out_tensor_tmp = makeTensor(out);
    }
    input_tensor = dataTypeCast(ctx, input_tensor, out_tensor_tmp.dtype());
    other_tensor = dataTypeCast(ctx, other_tensor, out_tensor_tmp.dtype());

    DiopiTensorT bcast_input_tensor = broadcastHelper(ctx, input_tensor, out_tensor_tmp);
    DiopiTensorT bcast_other_tensor = broadcastHelper(ctx, other_tensor, out_tensor_tmp);

    CnnlTensorDesc bcast_input_desc(bcast_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc bcast_other_desc(bcast_other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    cnnlTensorDescriptor_t input_descs[] = {bcast_input_desc.get(), bcast_other_desc.get()};
    const void* inputs[] = {bcast_input_tensor.data(), bcast_other_tensor.data()};

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

    diopiTensorHandle_t other_t;
    diopiSize_t input_shape;
    DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
    DIOPI_CALL(diopiRequireTensor(ctx, &other_t, &input_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, other_t, other));
    auto other_t_tensor = makeTensor(other_t);
    CnnlTensorDesc other_t_desc(other_t_tensor, CNNL_LAYOUT_ARRAY);
    diopiMul(ctx, out, input, other_t);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiMulScalar(ctx, input, input, other);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl

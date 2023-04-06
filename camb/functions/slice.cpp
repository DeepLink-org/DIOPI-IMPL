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

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto index_tensor = DiopiTensor(index);
    auto out_tensor = DiopiTensor(out);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim = dim + input_tensor.dim();
    }
    DIOPI_CALLCNNL(cnnlIndexSelect(handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), outDesc.get(), out_tensor.data()));
    return diopiSuccess;
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad, diopiSize_t input_sizes,
                                      int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = {diopi_dtype_int64, 0};
    diopiFill(ctx, grad_input, &zero);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_tensor = DiopiTensor(grad);
    CnnlTensorDesc grad_inputDesc(grad_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(grad_tensor, CNNL_LAYOUT_ARRAY);

    auto index_tensor = DiopiTensor(index);
    if (index_tensor.dtype() == diopi_dtype_int64) {
        dataTypeCast(ctx, index_tensor, diopi_dtype_int32);
    }
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim = dim + input_sizes.len;
    }
    DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                dim,
                                grad_inputDesc.get(),
                                grad_input_tensor.data(),
                                indexDesc.get(),
                                index_tensor.data(),
                                gradDesc.get(),
                                grad_tensor.data(),
                                grad_inputDesc.get(),
                                grad_input_tensor.data()));

    return diopiSuccess;
}

diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    diopiScalar_t index_scalar;
    index_scalar.stype = diopi_dtype_int64;
    index_scalar.ival = index;
    DiopiTensor index_tensor;
    makeTensorFromScalar(ctx, &index_scalar, index_tensor);
    auto out_tensor = DiopiTensor(out);

    if (dim < 0) {
        dim = dim + input_tensor.dim();
    }
    std::vector<int64_t> shape(out_tensor.shape());
    shape.insert(shape.begin() + dim, 1);
    out_tensor.reshape(shape);

    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlIndexSelect(handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), outDesc.get(), out_tensor.data()));
    return diopiSuccess;
}

diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                 int64_t dim, int64_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = {diopi_dtype_int64, 0};
    diopiFill(ctx, grad_input, &zero);
    auto grad_input_tensor = DiopiTensor(grad_input);
    CnnlTensorDesc grad_inputDesc(grad_input_tensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim = dim + input_sizes.len;
    }

    auto grad_tensor = DiopiTensor(grad_output);
    std::vector<int64_t> shape(grad_tensor.shape());
    shape.insert(shape.begin() + dim, 1);
    grad_tensor.reshape(shape);
    CnnlTensorDesc gradDesc(grad_tensor, CNNL_LAYOUT_ARRAY);

    diopiScalar_t index_scalar;
    index_scalar.stype = diopi_dtype_int64;
    index_scalar.ival = index;
    DiopiTensor index_tensor;
    makeTensorFromScalar(ctx, &index_scalar, index_tensor);
    if (index_tensor.dtype() == diopi_dtype_int64) {
        dataTypeCast(ctx, index_tensor, diopi_dtype_int32);
    }
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                dim,
                                grad_inputDesc.get(),
                                grad_input_tensor.data(),
                                indexDesc.get(),
                                index_tensor.data(),
                                gradDesc.get(),
                                grad_tensor.data(),
                                grad_inputDesc.get(),
                                grad_input_tensor.data()));

    return diopiSuccess;
}

// !
diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim,
                                int64_t index) {
    return diopiSuccess;
}

// !
diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim,
                               int64_t start, int64_t end, int64_t step) {
    return diopiSuccess;
}

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end,
                        int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(null_out);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> start_32(input_tensor.dim(), 0);
    std::vector<int32_t> step_32(input_tensor.dim(), 1);
    std::vector<int32_t> end_32(input_tensor.shape().begin(), input_tensor.shape().end());
    start_32[dim] = start;
    step_32[dim] = step;
    end_32[dim] = end;

    DIOPI_CALLCNNL(
        cnnlStridedSlice(handle, inputDesc.get(), input_tensor.data(), start_32.data(), end_32.data(), step_32.data(), outDesc.get(), out_tensor.data()));
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                int64_t dim, int64_t start, int64_t end, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(grad_output);
    auto out_tensor = DiopiTensor(grad_input);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> start_32(input_tensor.dim(), 0);
    std::vector<int32_t> step_32(input_tensor.dim(), 1);
    std::vector<int32_t> end_32(input_tensor.shape().begin(), input_tensor.shape().end());
    start_32[dim] = start;
    step_32[dim] = step;
    end_32[dim] = end;

    DIOPI_CALLCNNL(cnnlStridedSliceBackward(
        handle, start_32.data(), end_32.data(), step_32.data(), inputDesc.get(), input_tensor.data(), outDesc.get(), out_tensor.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

#include <diopi/functions.h>
#include <vector>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
namespace {

int getDim(int64_t dim, int64_t input_dim) {
    int dim_ = static_cast<int>(dim);
    if (dim_ < 0) {
        dim_ = dim_ + input_dim;
    }
    return dim_;
}

std::vector<uint32_t> getStride(int64_t dim, int64_t input_dim, int64_t step, const int64_t* input_sizes) {
    auto dim_ = getDim(dim, input_dim);
    std::vector<int64_t> stride;
    // fake stride for contiguous input
    int64_t z = 1;
    stride.insert(stride.begin(), z);
    for (int64_t d = input_dim - 1; d > 0; d--) {
        z *= input_sizes[d];
        stride.insert(stride.begin(), z);
    }

    std::vector<int64_t> new_stride(input_dim + 1);
    new_stride[input_dim] = input_dim == 0 ? 1 : stride[dim_];
    for (int64_t d = 0; d < input_dim; d++) {
        auto input_stride = stride[d];
        if (d == dim_) {
            new_stride[d] = step * input_stride;
        } else {
            new_stride[d] = input_stride;
        }
    }

    std::vector<uint32_t> strides;
    for (auto s : new_stride) {
        strides.emplace_back(static_cast<uint32_t>(s));
    }
    return strides;
}

}  // namespace

extern "C" {

DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECKCNNL(cnnlUnfold(handle,
                               static_cast<int>(dim),
                               static_cast<int>(size),
                               static_cast<int>(step),
                               input_desc.get(),
                               input_tensor.data(),
                               out_desc.get(),
                               out_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto grad_output_tensor = DiopiTensor(grad_output);
    auto grad_input_tensor = DiopiTensor(grad_input);

    CnnlTensorDesc grad_output_desc(grad_output_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_input_desc(grad_input_tensor, CNNL_LAYOUT_ARRAY);

    uint32_t workspace_size = 0;
    DIOPI_CHECKCNNL(cnnlGetAsStridedBackwardWorkspaceSize(handle, grad_input_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    auto strides = getStride(dim, input_sizes.len, step, input_sizes.data);

    DIOPI_CHECKCNNL(cnnlAsStridedBackward(handle,
                                          grad_output_desc.get(),
                                          grad_output_tensor.data(),
                                          grad_input_desc.get(),
                                          grad_input_tensor.data(),
                                          strides.data(),
                                          0,
                                          workspace,
                                          workspace_size));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

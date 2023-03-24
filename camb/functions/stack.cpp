#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {
extern "C" {
DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    std::vector<CnnlTensorDesc> inputsDesc(numTensors);
    std::vector<cnnlTensorDescriptor_t> inputs_desc(numTensors);
    std::vector<const void*> inputs_data(numTensors);

    // add a new dim for input_tensors
    for (int i = 0; i < numTensors; i++) {
        auto temp_tensor = DiopiTensor(tensors[i]);
        std::vector<int64_t> cat_shape = temp_tensor.shape();
        if (dim == -1) {
            dim = temp_tensor.shape().size();
        }
        cat_shape.insert(cat_shape.begin() + dim, 1);
        temp_tensor.reshape(cat_shape);
        inputsDesc[i].set(temp_tensor, CNNL_LAYOUT_ARRAY);
        inputs_desc[i] = inputsDesc[i].get();
        inputs_data[i] = temp_tensor.data();
    }
    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetConcatWorkspaceSize(handle, numTensors, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    auto out_tensor = DiopiTensor(out);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlConcat(handle, numTensors, dim, inputs_desc.data(), inputs_data.data(), workspace, workspace_size, out_desc.get(), out_tensor.data()));
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl
#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx,
                                 diopiTensorHandle_t values,
                                 diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input,
                                 int64_t k,
                                 int64_t dim,
                                 bool largest,
                                 bool sorted) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto indices_tensor = DiopiTensor(indices);
    auto values_tensor = DiopiTensor(values);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc values_desc(values_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indices_desc(indices_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetTopKTensorWorkspaceSize(handle, input_desc.get(), k, dim, largest, values_desc.get(), indices_desc.get(), &workspace_size));
    void *workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    //
    const bool lower_index_first = true;
    DIOPI_CALLCNNL(cnnlTopKTensor_v3(handle,
                                     input_desc.get(),
                                     input_tensor.data(),
                                     k,
                                     dim,
                                     largest,
                                     sorted,
                                     lower_index_first,
                                     workspace,
                                     workspace_size,
                                     values_desc.get(),
                                     values_tensor.data(),
                                     indices_desc.get(),
                                     indices_tensor.data()))
}

}  // extern "C"
}  // namespace camb
}  // namespace impl

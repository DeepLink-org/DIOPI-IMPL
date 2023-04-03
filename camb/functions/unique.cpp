#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlUniqueDescriptor_t, cnnlCreateUniqueDescriptor, cnnlDestroyUniqueDescriptor> uniqueDesc;
    cnnlUniqueDescriptor_t unique_desc = uniqueDesc.get();
    cnnlUniqueSort_t mode = sorted ? CNNL_UNSORT_FORWARD : CNNL_SORT_ASCEND;
    bool return_indices = true;
    // Currently, only the unique of the flattened input is supported.
    DIOPI_CHECK(*dim == -1, "Currently, only the unique of the flattened input is supported.")
    std::cout << "FLAG 1" << std::endl;
    DIOPI_CALLCNNL(cnnlSetUniqueDescriptor(unique_desc, mode, *dim, return_indices, return_counts));
    std::cout << "FLAG 2" << std::endl;
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetUniqueWorkSpace(handle, unique_desc, inputDesc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    std::cout << "FLAG 3" << std::endl;

    int* output_len = nullptr;
    DIOPI_CALLCNNL(cnnlUniqueGetOutLen(handle, unique_desc, inputDesc.get(), input_tensor.data(), workspace, output_len));
    std::cout << "FLAG 4" << std::endl;
    int64_t temp = *output_len; // something wrong here !!!!!
    std::cout << "FLAG 5" << std::endl;

    std::vector<int64_t> output_shape(1, temp);  

    auto output_tensor = requiresTensor(ctx, output_shape, input_tensor.dtype());
    std::vector<int64_t> counts_shape = {output_shape};
    auto counts_tensor = requiresTensor(ctx, counts_shape, diopi_dtype_int32);
    auto index_tensor = DiopiTensor(indices);

    DIOPI_CALLCNNL(cnnlUnique(handle,
                              unique_desc,
                              inputDesc.get(),
                              input_tensor.data(),
                              *output_len,
                              workspace,
                              output_tensor.data(),
                              (int*)index_tensor.data(),
                              (int*)counts_tensor.data()));
    *out = diopiTensorHandle_t(output_tensor);
    if (return_counts) {
        *counts = diopiTensorHandle_t(counts_tensor);
    }
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
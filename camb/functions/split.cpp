/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs, diopiConstTensorHandle_t input,
                                            const diopiSize_t splitSizes, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<CnnlTensorDesc> output_descs(num_outs);
    cnnlTensorDescriptor_t* desc_ptrs = new cnnlTensorDescriptor_t[num_outs];
    void** data_ptrs = new void*[num_outs];
    for (int i = 0; i < num_outs; ++i) {
        auto tensor = DiopiTensor(outs[i]);
        DIOPI_CALL(output_descs[i].set(tensor, CNNL_LAYOUT_ARRAY));
        desc_ptrs[i] = output_descs[i].get();
        data_ptrs[i] = tensor.data();
    }

    size_t worksapce_size;
    DIOPI_CALLCNNL(cnnlGetSplitWorkspaceSize(handle, num_outs, &worksapce_size));

    void* worksapce = nullptr;
    if (worksapce_size != 0) {
        worksapce = requiresBuffer(ctx, worksapce_size).data();
    }

    DIOPI_CALLCNNL(cnnlSplit(handle, num_outs, dim, input_desc.get(), input_tensor.data(), worksapce, worksapce_size, desc_ptrs, data_ptrs));

    delete[] desc_ptrs;
    delete[] data_ptrs;
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl

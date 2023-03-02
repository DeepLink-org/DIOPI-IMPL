#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"

namespace {
std::vector<int> getAxis(diopiConstTensorHandle_t input, diopiSize_t dim) {
    std::vector<int> axis;
    if (dim.len > 0) {
        for (int i = 0; i < dim.len; i++) {
          axis.push_back(static_cast<int>(dim.data[i]));
        }
    } else {
        auto input_tensor = impl::camb::makeTensor(input);
        int input_size = input_tensor.shape().size();
        for (int i = 0; i < input_size; i++) {
            axis.push_back(i);
        }
    }
    
    return axis;
}
} // namespace

extern "C" {

DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    auto stream = impl::camb::getStream(ctx);
    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    auto input_tensor = impl::camb::makeTensor(input);
    auto out_tensor = impl::camb::makeTensor(out);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc;
    if (dim.len > 0 && dim.len != input_tensor.shape().size()) {
        DIOPI_CALL(out_desc.set(out_tensor, CNNL_LAYOUT_ARRAY));
    } else {
        std::vector<int> out_dims = {1};
        DIOPI_CALL(out_desc.set(out_tensor, CNNL_LAYOUT_ARRAY, out_dims));
    }
    const void* input_ptr = input_tensor.data();
    void* out_ptr = out_tensor.data();

    CnnlResourceGuard<cnnlReduceDescriptor_t,
                      cnnlCreateReduceDescriptor,
                      cnnlDestroyReduceDescriptor>
        CnnlReduceDesc;
    cnnlReduceDescriptor_t reduce_desc = CnnlReduceDesc.get();
    cnnlDataType_t cnnl_dtype;
    DIOPI_CALL(convertType(&cnnl_dtype, dtype));
    std::vector<int> axis = getAxis(input, dim);
    DIOPI_CALLCNNL(cnnlSetReduceDescriptor(reduce_desc, axis.data(), axis.size(), CNNL_REDUCE_AVG,
                                           cnnl_dtype, CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES));
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, input_desc.get(), out_desc.get(), reduce_desc, &workspace_size));
    void *workspace = nullptr;
    if (0 != workspace_size) {
        workspace = impl::camb::requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlReduce(handle, reduce_desc, workspace, workspace_size,
                              nullptr, input_desc.get(), input_ptr, 0,
                              nullptr, nullptr, out_desc.get(), out_ptr));
}
}

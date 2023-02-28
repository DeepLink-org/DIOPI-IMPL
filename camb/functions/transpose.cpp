#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"

std::vector<int> GetPerm(diopiConstTensorHandle_t tensor_handle,
                         int64_t dim0,
                         int64_t dim1) {
    auto tensor = impl::camb::makeTensor(tensor_handle);
    diopiSize_t shape = tensor.shape();
    int input_size_ = shape.len;

    int dim0_ = 0;
    dim0_ = static_cast<int>(dim0);
    if (dim0_ < 0) {
        dim0_ = dim0_ + input_size_;
    }

    int dim1_ = 0;
    dim1_ = static_cast<int>(dim1);
    if (dim1_ < 0) {
        dim1_ = dim1_ + input_size_;
    }

    std::vector<int> perms(input_size_);
    std::iota(perms.begin(), perms.end(), 0);

    perms[dim0_] = dim1_;
    perms[dim1_] = dim0_;

    return perms;
}

extern "C" {

DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx,
                                      diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input,
                                      int64_t dim0,
                                      int64_t dim1) {
    auto stream = impl::camb::getStream(ctx);
    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTransposeDescriptor_t,
                      cnnlCreateTransposeDescriptor,
                      cnnlDestroyTransposeDescriptor>
        CnnlTransposeDesc;
    cnnlTransposeDescriptor_t transpose_desc = CnnlTransposeDesc.get();
    std::vector<int> perms = GetPerm(input, dim0, dim1);
    DIOPI_CALLCNNL(
        cnnlSetTransposeDescriptor(transpose_desc, perms.size(), perms.data()));

    auto input_tensor = impl::camb::makeTensor(input);
    auto output_tensor = impl::camb::makeTensor(out);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);
    const void* input_ptr = input_tensor.data();
    void* out_ptr = output_tensor.data();

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(
        handle, input_desc.get(), transpose_desc, &workspace_size));
    void* workspace =
        workspace_size == 0
            ? impl::camb::requiresBuffer(ctx, workspace_size).data()
            : nullptr;
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle,
                                    transpose_desc,
                                    input_desc.get(),
                                    input_ptr,
                                    output_desc.get(),
                                    out_ptr,
                                    workspace,
                                    workspace_size));
    return diopiSuccess;
}
}
#include <numeric>
#include <string.h>
#include "../helper.hpp"

namespace {
diopiError_t InitTensorDesc(diopiConstTensorHandle_t tensor_handle, cnnlTensorDescriptor_t &desc){
    auto tensor = impl::camb::makeTensor(tensor_handle);
    std::vector<int> src_shape = tensor.shape();
    int dimNb = tensor.shape_len();
    std::vector<int> shape(dimNb);

    if (dimNb == 0) {
        dimNb = 1;
        shape.push_back(1);
    } else {
	shape = src_shape;
    }
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, tensor.dtype()));
    if(CNNL_DTYPE_DOUBLE == dtype){
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_ARRAY, dtype, dimNb, shape.data()));
}

diopiError_t InitTensor(diopiTensorHandle_t tensor_handle, cnnlTensorDescriptor_t &desc, void* &ptr){
    DIOPI_CALL(InitTensorDesc(tensor_handle, desc));
    auto tensor = impl::camb::makeTensor(tensor_handle);
    ptr = tensor.data();
    return diopiSuccess;
}

diopiError_t InitTensor(diopiConstTensorHandle_t tensor_handle, cnnlTensorDescriptor_t &desc,const void* &ptr){
    DIOPI_CALL(InitTensorDesc(tensor_handle, desc));
    auto tensor = impl::camb::makeTensor(tensor_handle);
    ptr = tensor.data();
    return diopiSuccess;
}
}

std::vector<int> GetPerm(diopiConstTensorHandle_t tensor_handle, int64_t dim0, int64_t dim1) {
    auto tensor = impl::camb::makeTensor(tensor_handle);
    std::vector<int> src_shape = tensor.shape();
    int input_size_ = tensor.shape_len();

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

DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {

    HandleGuard handle_guard;
    auto stream  = impl::camb::getStream(ctx);
    cnnlHandle_t handle = handle_guard.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> CnnlTransposeDesc;
    cnnlTransposeDescriptor_t transpose_desc = CnnlTransposeDesc.get();

    std::vector<int> perms = GetPerm(input, dim0, dim1);
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(transpose_desc, perms.size(), perms.data()));

    TensorDescGuard x_desc_guard, y_desc_guard;
    cnnlTensorDescriptor_t input_desc = x_desc_guard.get();
    cnnlTensorDescriptor_t output_desc = y_desc_guard.get();
    const void* input_ptr;
    void* output_ptr;
    DIOPI_CALL(InitTensor(input, input_desc, input_ptr));
    DIOPI_CALL(InitTensor(out, output_desc, output_ptr));

    size_t workspace_size = 0;
    void *workspace = nullptr;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, input_desc, transpose_desc, &workspace_size));
    if (workspace_size != 0) {
        cnrtMalloc((void **)&(workspace), workspace_size);
    }
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transpose_desc, input_desc, input_ptr, output_desc, output_ptr, workspace, workspace_size));
}

}

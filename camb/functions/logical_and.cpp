#include <vector>

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

extern "C" {
DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx,
                                       diopiTensorHandle_t out,
                                       diopiConstTensorHandle_t input,
                                       diopiConstTensorHandle_t other) {
    auto stream = impl::camb::getStream(ctx);

    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    diopiTensorHandle_t other_ = diopiTensorHandle_t(other);

    auto trInput = impl::camb::makeTensor(input_);
    auto trOther = impl::camb::makeTensor(other_);
    auto trOut = impl::camb::makeTensor(out);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    cnnlTensorLayout_t input_layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t input_dtype;
    DIOPI_CALL(convertType(&input_dtype, trInput.dtype()));
    cnnlTensorLayout_t other_layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t other_dtype;
    DIOPI_CALL(convertType(&other_dtype, trOther.dtype()));
    cnnlTensorLayout_t out_layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t out_dtype;
    DIOPI_CALL(convertType(&out_dtype, trOut.dtype()));

    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> InputCnnlDesc;
    cnnlTensorDescriptor_t input_desc = InputCnnlDesc.get();
    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> OtherCnnlDesc;
    cnnlTensorDescriptor_t other_desc = OtherCnnlDesc.get();
    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> OutCnnlDesc;
    cnnlTensorDescriptor_t out_desc = OutCnnlDesc.get();

    diopiSize_t input_shape = trInput.shape();
    int input_dimNb = input_shape.len;
    std::vector<int> input_dimSize(input_dimNb);
    if (input_dimNb == 0) {
        input_dimNb = 1;
        input_dimSize.push_back(1);
    } else {
        for (int i = 0; i < input_dimNb; ++i) {
            input_dimSize[i] = input_shape.data[i];
        }
    }

    diopiSize_t other_shape = trOther.shape();
    int other_dimNb = other_shape.len;
    std::vector<int> other_dimSize(other_dimNb);
    if (other_dimNb == 0) {
        other_dimNb = 1;
        other_dimSize.push_back(1);
    } else {
        for (int i = 0; i < other_dimNb; ++i) {
            other_dimSize[i] = other_shape.data[i];
        }
    }

    diopiSize_t out_shape = trOut.shape();
    int out_dimNb = out_shape.len;
    std::vector<int> out_dimSize(out_dimNb);
    if (out_dimNb == 0) {
        out_dimNb = 1;
        out_dimSize.push_back(1);
    } else {
        for (int i = 0; i < out_dimNb; ++i) {
            out_dimSize[i] = out_shape.data[i];
        }
    }

    if (trInput.numel() < trOut.numel() || trOther.numel() < trOut.numel()) {
        input_dtype = CNNL_DTYPE_BOOL;
        other_dtype = CNNL_DTYPE_BOOL;

        if (input_dimNb < out_dimNb) {
            input_dimSize.insert(input_dimSize.begin(), out_dimNb - input_dimNb, 1);
            input_dimNb = out_dimNb;
        }

        if (other_dimNb < out_dimNb) {
            other_dimSize.insert(other_dimSize.begin(), out_dimNb - other_dimNb, 1);
            other_dimNb = out_dimNb;
        }

        cnnlSetTensorDescriptor(input_desc, input_layout, input_dtype, input_dimNb, input_dimSize.data());
        cnnlSetTensorDescriptor(other_desc, other_layout, other_dtype, other_dimNb, other_dimSize.data());
        cnnlSetTensorDescriptor(out_desc, out_layout, out_dtype, out_dimNb, out_dimSize.data());

        diopiSize_t expand_input_size = diopiSize_t_(out_shape.data, out_dimNb);
        diopiSize_t expand_other_size = diopiSize_t_(out_shape.data, out_dimNb);

        auto expand_input = impl::camb::requiresTensor(ctx, expand_input_size, diopi_dtype_bool);
        cnnlExpand(handle, input_desc, trInput.data(), out_desc, expand_input.data());
        input_dimSize = out_dimSize;

        auto expand_other = impl::camb::requiresTensor(ctx, expand_other_size, diopi_dtype_bool);
        cnnlExpand(handle, other_desc, trOther.data(), out_desc, expand_other.data());
        other_dimSize = out_dimSize;

        cnnlSetTensorDescriptor(input_desc, input_layout, input_dtype, input_dimNb, input_dimSize.data());
        cnnlSetTensorDescriptor(other_desc, other_layout, other_dtype, other_dimNb, other_dimSize.data());
        // cnnlSetTensorDescriptor(out_desc, out_layout, out_dtype, out_dimNb, out_dimSize.data());

        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc, other_desc, out_desc, &workspace_size));
        cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
        diopiTensorHandle_t workspace;
        diopiRequireBuffer(ctx, &workspace, workspace_size, diopi_device);
        DIOPI_CALLCNNL(cnnlLogicOp(handle,
                                   logic_op_and,
                                   input_desc,
                                   expand_input.data(),
                                   other_desc,
                                   expand_other.data(),
                                   &workspace,
                                   workspace_size,
                                   out_desc,
                                   trOut.data()));
    } else {
        cnnlSetTensorDescriptor(input_desc, input_layout, input_dtype, input_dimNb, input_dimSize.data());
        cnnlSetTensorDescriptor(other_desc, other_layout, other_dtype, other_dimNb, other_dimSize.data());
        cnnlSetTensorDescriptor(out_desc, out_layout, out_dtype, out_dimNb, out_dimSize.data());

        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc, other_desc, out_desc, &workspace_size));
        cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
        diopiTensorHandle_t workspace;
        diopiRequireBuffer(ctx, &workspace, workspace_size, diopi_device);
        DIOPI_CALLCNNL(cnnlLogicOp(handle,
                                   logic_op_and,
                                   input_desc,
                                   trInput.data(),
                                   other_desc,
                                   trOther.data(),
                                   &workspace,
                                   workspace_size,
                                   out_desc,
                                   trOut.data()));
    }
}
}  // extern "C"
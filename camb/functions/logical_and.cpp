#include <diopi/functions.h>
#include <vector>
#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {
DIOPI_API diopiError_t castToBool(cnnlHandle_t handle,
                                  const cnnlTensorDescriptor_t inputDesc,
                                  diopiDtype_t inputDtype,
                                  const void* inputData,
                                  const cnnlTensorDescriptor_t outDesc,
                                  void* outData) {
    cnnlCastDataType_t cast_type;
    switch (inputDtype) {
        case diopi_dtype_float16:
            cast_type = CNNL_CAST_HALF_TO_BOOL;
            break;
        case diopi_dtype_float32:
            cast_type = CNNL_CAST_FLOAT_TO_BOOL;
            break;
        case diopi_dtype_int32:
            cast_type = CNNL_CAST_INT32_TO_BOOL;
            break;
    }
    cnnlCastDataType(handle, inputDesc, inputData, cast_type, outDesc, outData);
    return diopiSuccess;
}

DIOPI_API diopiError_t castFromBool(cnnlHandle_t handle,
                                    const cnnlTensorDescriptor_t inputDesc,
                                    const void* inputData,
                                    diopiDtype_t outDtype,
                                    const cnnlTensorDescriptor_t outDesc,
                                    void* outData) {
    cnnlCastDataType_t cast_type;
    switch (outDtype) {
        case diopi_dtype_float16:
            cast_type = CNNL_CAST_BOOL_TO_HALF;
            break;
        case diopi_dtype_float32:
            cast_type = CNNL_CAST_BOOL_TO_FLOAT;
            break;
        case diopi_dtype_int32:
            cast_type = CNNL_CAST_BOOL_TO_INT32;
            break;
    }
    DIOPI_CALLCNNL(cnnlCastDataType(handle, inputDesc, inputData, cast_type, outDesc, outData));
}

DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t input, diopiTensorHandle_t other) {
    auto stream = impl::camb::getStream(ctx);
    auto trInput = impl::camb::makeTensor(input);
    auto trOther = impl::camb::makeTensor(other);
    auto trOut = impl::camb::makeTensor(out);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));
    CnnlTensorDesc inputDesc(trInput, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(trOther, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(trOut, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc boolInputDesc(trInput, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc boolOtherDesc(trOther, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc boolOutDesc(trOut, CNNL_LAYOUT_ARRAY);

    cnnlDataType_t inputDtype;
    cnnlDataType_t otherDtype;
    cnnlDataType_t outDtype;
    CnnlDataType::convertToCnnlType(&inputDtype, trInput.dtype());
    CnnlDataType::convertToCnnlType(&otherDtype, trOther.dtype());
    CnnlDataType::convertToCnnlType(&outDtype, trOut.dtype());

    std::vector<int32_t> input_dimSize = trInput.shape();
    int input_dimNb = input_dimSize.size();

    std::vector<int32_t> other_dimSize = trOther.shape();
    int other_dimNb = other_dimSize.size();

    std::vector<int32_t> out_dimSize = trOut.shape();
    int out_dimNb = out_dimSize.size();

    cnnlSetTensorDescriptor(inputDesc.get(), CNNL_LAYOUT_ARRAY, inputDtype, input_dimNb, input_dimSize.data());
    cnnlSetTensorDescriptor(otherDesc.get(), CNNL_LAYOUT_ARRAY, otherDtype, other_dimNb, other_dimSize.data());
    cnnlSetTensorDescriptor(outDesc.get(), CNNL_LAYOUT_ARRAY, outDtype, out_dimNb, out_dimSize.data());

    cnnlSetTensorDescriptor(boolInputDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, input_dimNb, input_dimSize.data());
    cnnlSetTensorDescriptor(boolOtherDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, other_dimNb, other_dimSize.data());
    cnnlSetTensorDescriptor(boolOutDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());

    void* boolInputData = nullptr;
    if (trInput.dtype() != diopi_dtype_bool) {
        boolInputData = impl::camb::requiresBuffer(ctx, trInput.numel()).data();
        castToBool(handle, inputDesc.get(), trInput.dtype(), trInput.data(), boolInputDesc.get(), boolInputData);
    } else {
        boolInputData = trInput.data();
    }

    void* boolOtherData = nullptr;
    if (trOther.dtype() != diopi_dtype_bool) {
        boolOtherData = impl::camb::requiresBuffer(ctx, trOther.numel()).data();
        castToBool(handle, otherDesc.get(), trOther.dtype(), trOther.data(), boolOtherDesc.get(), boolOtherData);
    } else {
        boolOtherData = trOther.data();
    }

    void* boolOutData = nullptr;
    if ((trOut.dtype() != diopi_dtype_bool)) {
        boolOutData = impl::camb::requiresBuffer(ctx, trOut.numel()).data();
        castToBool(handle, outDesc.get(), trOut.dtype(), trOut.data(), boolOutDesc.get(), boolOutData);
    } else {
        boolOutData = trOut.data();
    }

    int out_numel = 1;
    for (int i = 0; i < out_dimNb; i++) {
        out_numel *= out_dimSize[i];
    }

    if (trInput.numel() < out_numel || trOther.numel() < out_numel) {
        auto boolInput = impl::camb::requiresBuffer(ctx, out_numel);
        cnnlExpand(handle, boolInputDesc.get(), boolInputData, boolOutDesc.get(), boolInput.data());
        boolInputData = boolInput.data();

        auto boolOther = impl::camb::requiresBuffer(ctx, out_numel);
        cnnlExpand(handle, boolOtherDesc.get(), boolOtherData, boolOutDesc.get(), boolOther.data());
        boolOtherData = boolOther.data();

        cnnlSetTensorDescriptor(boolInputDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());
        cnnlSetTensorDescriptor(boolOtherDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());
        cnnlSetTensorDescriptor(boolOutDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, boolInputDesc.get(), boolOtherDesc.get(), boolOutDesc.get(), &workspace_size));
    cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
    diopiTensorHandle_t workspace;
    diopiRequireBuffer(ctx, &workspace, workspace_size, diopi_device);

    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logic_op_and,
                               boolInputDesc.get(),
                               boolInputData,
                               boolOtherDesc.get(),
                               boolOtherData,
                               &workspace,
                               workspace_size,
                               boolOutDesc.get(),
                               boolOutData));
    if (input = out) {
        castFromBool(handle, boolOutDesc.get(), boolOutData, trInput.dtype(), inputDesc.get(), trInput.data());
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t other) {
    diopiLogicalAnd(ctx, input, input, other);
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl

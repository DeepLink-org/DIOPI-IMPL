#include <diopi/functions.h>
#include <iostream>
#include <vector>
#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {
DIOPI_API diopiError_t castToBool(
    cnnlHandle_t handle, const cnnlTensorDescriptor_t inputDesc, 
    diopiDtype_t inputDtype, const void* inputData, 
    const cnnlTensorDescriptor_t outDesc, void* outData) {
    cnnlCastDataType_t cast_type;
    switch (inputDtype) {
        case diopi_dtype_float16:
            std::cout << "FLAG1" << std::endl;
            cast_type = CNNL_CAST_HALF_TO_BOOL;
            break;
        case diopi_dtype_float32:
            std::cout << "FLAG2" << std::endl;
            cast_type = CNNL_CAST_FLOAT_TO_BOOL;
            break; 
        case diopi_dtype_int32:
            std::cout << "FLAG3" << std::endl;
            cast_type = CNNL_CAST_INT32_TO_BOOL;
            break;
    }
    cnnlCastDataType(handle, inputDesc, inputData, cast_type, outDesc, outData); 
    std::cout << "cast to bool succeed!!" << std::endl;
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
            diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    auto stream = impl::camb::getStream(ctx);
    auto trInput = impl::camb::makeTensor(input);
    auto trOther = impl::camb::makeTensor(other);
    auto trOut = impl::camb::makeTensor(out);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));
    // 原始描述符
    CnnlTensorDesc inputDesc(trInput, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(trOther, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(trOut, CNNL_LAYOUT_ARRAY);
    // 类型转换为BOOL后的描述符
    CnnlTensorDesc boolInputDesc(trInput, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc boolOtherDesc(trOther, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc boolOutDesc(trOut, CNNL_LAYOUT_ARRAY);

    cnnlDataType_t inputDtype;
    cnnlDataType_t otherDtype;
    cnnlDataType_t outDtype;
    CnnlDataType::convertToCnnlType(&inputDtype, trInput.dtype());
    CnnlDataType::convertToCnnlType(&otherDtype, trOther.dtype());
    CnnlDataType::convertToCnnlType(&outDtype, trOut.dtype());

    std::vector<int32_t> input_shape = trInput.shape();
    int input_dimNb = input_shape.size();
    std::vector<int32_t> input_dimSize(input_dimNb);
    if (input_dimNb == 0) {
        input_dimNb = 1;
        input_dimSize.push_back(1);
    } else {
        std::cout << "input_dimSize: " << std::endl;
        for (int i = 0; i < input_dimNb; i++) {
            input_dimSize[i] = input_shape.data()[i];
            std::cout << input_dimSize[i] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<int32_t> other_shape = trOther.shape();
    int other_dimNb = other_shape.size();
    std::vector<int32_t> other_dimSize(other_dimNb);
    if (other_dimNb == 0) {
        other_dimNb = 1;
        other_dimSize.push_back(1);
    } else {
        std::cout << "other_dimSize: " << std::endl;
        for (int i = 0; i < other_dimNb; i++) {
            other_dimSize[i] = other_shape.data()[i];
            std::cout << other_dimSize[i] << " ";
        }
        std::cout << std::endl;
    }

    // 比较input和other的维度数，不足的维度用1补齐
    int max_dimNb = input_dimNb > other_dimNb ? input_dimNb : other_dimNb;
    if (input_dimNb < max_dimNb) {
        std::cout << "fill input_dimNb" << std::endl;
        input_dimSize.insert(input_dimSize.begin(), max_dimNb - input_dimNb, 1);
        input_dimNb = max_dimNb;
    }

    if (other_dimNb < max_dimNb) {
        std::cout << "fill other_dimNb" << std::endl;
        other_dimSize.insert(other_dimSize.begin(), max_dimNb - other_dimNb, 1);
        other_dimNb = max_dimNb;
    }

    int out_dimNb = max_dimNb;
    std::vector<int32_t> out_dimSize(out_dimNb);
    if (out_dimNb == 0) {
        out_dimNb = 1;
        out_dimSize.push_back(1);
    } else {
        std::cout << "out_dimSize: " << std::endl;
        for (int i = 0; i < out_dimNb; i++) {
            out_dimSize[i] = input_dimSize[i] > other_dimSize[i] ? input_dimSize[i] : other_dimSize[i];
            std::cout << out_dimSize[i] << " ";
        }
        std::cout << std::endl;
    }

    cnnlSetTensorDescriptor(inputDesc.get(), CNNL_LAYOUT_ARRAY, inputDtype, input_dimNb, input_dimSize.data());
    cnnlSetTensorDescriptor(otherDesc.get(), CNNL_LAYOUT_ARRAY, otherDtype, other_dimNb, other_dimSize.data());
    cnnlSetTensorDescriptor(outDesc.get(), CNNL_LAYOUT_ARRAY, outDtype, out_dimNb, out_dimSize.data());

    cnnlSetTensorDescriptor(boolInputDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, input_dimNb, input_dimSize.data());
    cnnlSetTensorDescriptor(boolOtherDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, other_dimNb, other_dimSize.data());
    cnnlSetTensorDescriptor(boolOutDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());
    

    void* boolInputData = nullptr;
    if (trInput.dtype() != diopi_dtype_bool) {
        std::cout << "ready to cast to bool!!!" << std::endl;
        std::cout << "cast from " << trInput.dtype() << std::endl;
        boolInputData = impl::camb::requiresBuffer(ctx, trInput.numel()).data();
        castToBool(handle, inputDesc.get(), trInput.dtype(), trInput.data(), boolInputDesc.get(), boolInputData);
    }
    else {
        std::cout << "no need to cast!!" << std::endl;
        boolInputData = const_cast<void*>(trInput.data());
    }
  
   void* boolOtherData = nullptr;
    if (trOther.dtype() != diopi_dtype_bool) {
        std::cout << "ready to cast to bool!!!" << std::endl;
        std::cout << "cast from " << trOther.dtype() << std::endl;
        boolOtherData = impl::camb::requiresBuffer(ctx, trOther.numel()).data();
        castToBool(handle, otherDesc.get(), trOther.dtype(), trOther.data(), boolOtherDesc.get(), boolOtherData);
    }
    else {
        std::cout << "no need to cast!!" << std::endl;
        boolOtherData = const_cast<void*>(trOther.data());
    }

    void* boolOutData = nullptr;
    if (trOut.dtype() != diopi_dtype_bool) {
        std::cout << "ready to cast to bool!!!" << std::endl;
        std::cout << "cast from " << trOut.dtype() << std::endl;
        boolOutData = impl::camb::requiresBuffer(ctx, trOut.numel()).data();
        castToBool(handle, outDesc.get(), trOut.dtype(), trOut.data(), boolOutDesc.get(), boolOutData);
    }
    else {
        std::cout << "no need to cast!!" << std::endl;
        boolOutData = const_cast<void*>(trOut.data());
    }

    int out_numel = 1;
    for (int i = 0; i < out_dimNb; i++) {
        out_numel *= out_dimSize[i];
    }
    std::cout << "out_numel: " << out_numel << std::endl;

    // 判断是否需要进行broadcast
    if (trInput.numel() < out_numel || trOther.numel() < out_numel) {
        std::cout << "ready for broadcast!!!!!" << std::endl;

        auto boolExpandInput = impl::camb::requiresBuffer(ctx, out_numel);
        cnnlExpand(handle, boolInputDesc.get(), boolInputData, boolOutDesc.get(), boolExpandInput.data());
       
        auto boolExpandOther = impl::camb::requiresBuffer(ctx, out_numel);
        cnnlExpand(handle, boolOtherDesc.get(), boolOtherData, boolOutDesc.get(), boolExpandOther.data());

        cnnlSetTensorDescriptor(
            boolInputDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());
        cnnlSetTensorDescriptor(
            boolOtherDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());
        cnnlSetTensorDescriptor(
            boolOutDesc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL, out_dimNb, out_dimSize.data());

        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, boolInputDesc.get(), boolOtherDesc.get(), boolOutDesc.get(), &workspace_size));
        cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
        diopiTensorHandle_t workspace;
        diopiRequireBuffer(ctx, &workspace, workspace_size, diopi_device);
        std::cout << "buffer allocated!!!!" << std::endl;
        std::cout << "workspace_size is " << workspace_size << std::endl;
        DIOPI_CALLCNNL(cnnlLogicOp(handle,
                                   logic_op_and,
                                   boolInputDesc.get(),
                                   boolExpandInput.data(),
                                   boolOtherDesc.get(),
                                   boolExpandOther.data(),
                                   &workspace,
                                   workspace_size,
                                   boolOutDesc.get(),
                                   boolOutData));
        return diopiSuccess;
    } else {
        std::cout << "no broadcast!!" << std::endl;

        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, boolInputDesc.get(), boolOtherDesc.get(), boolOutDesc.get(), &workspace_size));
        cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
        diopiTensorHandle_t workspace;
        diopiRequireBuffer(ctx, &workspace, workspace_size, diopi_device);
        std::cout << "buffer allocated!!!!" << std::endl;
        std::cout << "workspace_size is " << workspace_size << std::endl;

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
        return diopiSuccess;
    }
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

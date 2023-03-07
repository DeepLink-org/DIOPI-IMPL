#include <vector>

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

#include <iostream>


extern "C" {
DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx,
                                       diopiTensorHandle_t out,
                                       diopiConstTensorHandle_t input,
                                       diopiConstTensorHandle_t other) {
    auto stream = impl::camb::getStream(ctx);
    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    diopiTensorHandle_t other_ = diopiTensorHandle_t(other);

    auto trInput = impl::camb::makeTensor(input_);
    auto trOther = impl::camb::makeTensor(other_);
    auto trOut = impl::camb::makeTensor(out);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
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

    CnnlTensorDesc input_desc(trInput, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(trOther, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(trOut, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> input_shape = trInput.shape();
    int input_dimNb = input_shape.size();
    std::vector<int32_t> input_dimSize(input_dimNb);
    if (input_dimNb == 0) {
        input_dimNb = 1;
        input_dimSize.push_back(1);
    }
    else {
        std::cout << "input_dimSize: " << std::endl;
        for (int i=0; i < input_dimNb; i++) {
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
    }
    else {
        std::cout << "other_dimSize: " << std::endl;
        for (int i=0; i < other_dimNb; i++) {
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
    }
    else {
        std::cout << "out_dimSize: " << std::endl;
        for (int i=0; i < out_dimNb; i++) {
            out_dimSize[i] = input_dimSize[i] > other_dimSize[i] ? input_dimSize[i] : other_dimSize[i];
            std::cout << out_dimSize[i] << " ";
        }
        std::cout << std::endl;
    }

    int out_numel = 1;
        for (int i=0; i < out_dimNb; i++) {
            out_numel *= out_dimSize[i];
        }
    std::cout << "out_numel: " << out_numel << std::endl;

    // input_dtype = CNNL_DTYPE_BOOL;
    // other_dtype = CNNL_DTYPE_BOOL;

    // 判断是否需要进行broadcast
    if (trInput.numel() < out_numel || trOther.numel() < out_numel){
        std::cout << "ready for broadcast!!!!!" << std::endl;
        // 在expand之前将信息绑定到desc一次
        cnnlSetTensorDescriptor(
            input_desc.get(), input_layout, input_dtype, input_dimNb, input_dimSize.data());
        cnnlSetTensorDescriptor(
            other_desc.get(), other_layout, other_dtype, other_dimNb, other_dimSize.data());
        cnnlSetTensorDescriptor(
            out_desc.get(), out_layout, out_dtype, out_dimNb, out_dimSize.data());
        
        std::vector<int64_t> cnnl_out_shape(out_dimSize.begin(), out_dimSize.end());
        diopiSize_t target_size(cnnl_out_shape.data(), cnnl_out_shape.size());

        auto expand_input = impl::camb::requiresTensor(ctx, target_size, trInput.dtype());
        cnnlExpand(handle, input_desc.get(), trInput.data(), out_desc.get(), expand_input.data());
        std::cout << "expand_input.size() is " << expand_input.numel() << std::endl;
        std::cout << "expand succeed" << std::endl;

        auto expand_other = impl::camb::requiresTensor(ctx, target_size, trOther.dtype());
        cnnlExpand(handle, other_desc.get(), trOther.data(), out_desc.get(), expand_other.data());
        std::cout << "expand_other.size() is " << expand_other.numel() << std::endl;
        std::cout << "expand succeed" << std::endl;
        
        cnnlSetTensorDescriptor(
            input_desc.get(), input_layout, input_dtype, input_dimNb, input_dimSize.data());
        cnnlSetTensorDescriptor(
            other_desc.get(), other_layout, other_dtype, other_dimNb, other_dimSize.data());
        cnnlSetTensorDescriptor(
            out_desc.get(), out_layout, out_dtype, out_dimNb, out_dimSize.data());
        std::cout << "input_dimNb = " << input_dimNb << std::endl;
        std::cout << "other_dimNb = " << other_dimNb << std::endl;
        std::cout << "out_dimNb = " << out_dimNb << std::endl;

        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_desc.get(), out_desc.get(), &workspace_size));
        cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
        diopiTensorHandle_t workspace;
        diopiRequireBuffer(ctx, &workspace, workspace_size, diopi_device);
        std::cout << "buffer allocated!!!!" << std::endl;
        std::cout << "workspace_size is " << workspace_size << std::endl;
        DIOPI_CALLCNNL(cnnlLogicOp(handle,
                                logic_op_and,
                                input_desc.get(),
                                expand_input.data(),
                                other_desc.get(),
                                expand_other.data(),
                                &workspace,
                                workspace_size,
                                out_desc.get(),
                                trOut.data()));
        return diopiSuccess;
    }
        else {
        std::cout << "no broadcast!!" << std::endl;
        cnnlSetTensorDescriptor(
            input_desc.get(), input_layout, input_dtype, input_dimNb, input_dimSize.data());
        cnnlSetTensorDescriptor(
            other_desc.get(), other_layout, other_dtype, other_dimNb, other_dimSize.data());
        cnnlSetTensorDescriptor(
            out_desc.get(), out_layout, out_dtype, out_dimNb, out_dimSize.data());

        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_desc.get(), out_desc.get(), &workspace_size));
        cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
        diopiTensorHandle_t workspace;
        diopiRequireBuffer(ctx, &workspace, workspace_size, diopi_device);
        std::cout << "buffer allocated!!!!" << std::endl;
        std::cout << "workspace_size is " << workspace_size << std::endl;

        DIOPI_CALLCNNL(cnnlLogicOp(handle,
                                logic_op_and,
                                input_desc.get(),
                                trInput.data(),
                                other_desc.get(),
                                trOther.data(),
                                &workspace,
                                workspace_size,
                                out_desc.get(),
                                trOut.data()));
        return diopiSuccess;
        }
}
}  // extern "C"
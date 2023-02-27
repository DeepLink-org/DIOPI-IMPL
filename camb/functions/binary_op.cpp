#include <vector>
#include <iostream>

#include <diopi/functions.h>
#include "../cnnl_helper.hpp"
#include <cnrt.h>
extern "C" {

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha){

    auto stream  = impl::camb::getStream(ctx);
    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    diopiTensorHandle_t other_ = diopiTensorHandle_t(other);
    auto trInput = impl::camb::makeTensor(input_);
    auto trOther = impl::camb::makeTensor(other_);
    auto trOutput = impl::camb::makeTensor(out);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescInput;
    cnnlTensorDescriptor_t descInput = CnnlDescInput.get();

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescOther;
    cnnlTensorDescriptor_t descOther = CnnlDescOther.get();

     CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescOut;
    cnnlTensorDescriptor_t descOut = CnnlDescOut.get();

    int dimNbInput =trInput.shape().len;
    std::vector<int> dimSizeInput(dimNbInput);
    if (dimNbInput == 0) {
        dimNbInput = 1;
        dimSizeInput.push_back(1);
    } else {
        for (int i = 0; i < dimNbInput; ++i) {
            dimSizeInput[i] = trInput.shape().data[i];
        }
    }

    diopiSize_t shapeOther = trInput.shape();
    int dimNbOther = trOther.shape().len;
    std::vector<int> dimSizeOther(dimNbOther);
    if (dimNbOther == 0) {
        dimNbOther = 1;
        dimSizeOther.push_back(1);
    } else {
        for (int i = 0; i < dimNbOther; ++i) {
            dimSizeOther[i] = trOther.shape().data[i];
        }
    }

    int dimNbOut = trOutput.shape().len;
    std::vector<int> dimSizeOut(dimNbOut);
    if (dimNbOut == 0) {
        dimNbOut = 1;
        dimSizeOut.push_back(1);
    } else {
        for (int i = 0; i < dimNbOut; ++i) {
            dimSizeOut[i] = trOutput.shape().data[i];
        }
    }

    void *pAlphaIn = (void*)malloc(4);
    void *pBetaIn = (void*)malloc(4);
    if (dtype >= 3 && dtype <= 13){
        *(int32_t*)pBetaIn = 0;
        if(alpha->stype < 7){
            *(int32_t*)pAlphaIn = (int32_t)alpha->ival;
        }
        else{
            *(int32_t*)pAlphaIn = (int32_t)(float)alpha->fval;
        }
    }
    else{
        *(float*)pBetaIn = 0.0f;
        if(alpha->stype < 7){
            *(float*)pAlphaIn = (float)(int32_t)alpha->ival;
        }
        else{
            *(float*)pAlphaIn = (float)alpha->fval;
        }
    }

    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descInput, layout, dtype, dimNbInput, dimSizeInput.data()));
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descOther, layout, dtype, dimNbOther, dimSizeOther.data()));
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descOut, layout, dtype, dimNbOut, dimSizeOut.data()));

    DIOPI_CALLCNNL(cnnlTransform(handle, pAlphaIn, descOther, trOther.data(), pBetaIn, trOutput.data())); 
    free(pAlphaIn);
    free(pBetaIn);  
    const cnnlTensorDescriptor_t input_descs[2] = {descInput, descOther};
    const void* inputs[2] = {trInput.data(),trOther.data()};
    uint32_t input_num = 2;
    
    std::cout << "dimSizeInput: " << dimNbInput << "\t dimSizeOther: " << dimNbOther << "\t dimSizeOut: " <<  dimNbOut << std::endl;
    DIOPI_CALLCNNL(cnnlAddN(handle, input_descs, inputs, input_num, descOut, trOutput.data()));
    return diopiSuccess;


}




}  // extern "C"
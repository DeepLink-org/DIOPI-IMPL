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

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescInput;
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));
    cnnlTensorDescriptor_t descInput = CnnlDescInput.get();

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescOther;
    cnnlTensorDescriptor_t descOther = CnnlDescOther.get();
    diopiSize_t shape = trInput.shape();
    int dimNb = shape.len;
    std::vector<int> dimSize(dimNb);

    if (dimNb == 0) {
        dimNb = 1;
        dimSize.push_back(1);
    } else {
        for (int i = 0; i < dimNb; ++i) {
            dimSize[i] = shape.data[i];
        }
    }

    void *pAlphaIn = (void*)malloc(4);
    void *pBetaIn = (void*)malloc(4);
    if (alpha->stype < 7){
        *(float*)pAlphaIn = (float)(int32_t)alpha->ival;
    }
    else{
        *(float*)pAlphaIn = (float)alpha->fval;
    }
    *(float*)pBetaIn = 0.0f;


    // std::cout<<"other: "<<trOther.data()<<std::endl;
    std::cout<<"alphaIn: "<<(*(int32_t*)pAlphaIn)<<std::endl;
    // std::cout<<"input: "<<trInput.data()<<std::endl;

    // std::cout<<"Alpha: "<<*Alpha<<std::endl;
    // std::cout<<"Beta: "<<*Beta<<std::endl;

    // auto output = impl::camb::makeTensor(out);
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descInput, layout, dtype, dimNb, dimSize.data()));
    // std::cout<<"000"<<std::endl;
    DIOPI_CALLCNNL(cnnlTransform(handle, pAlphaIn, descInput, trOther.data(), pBetaIn, trOutput.data()));

    // std::cout<<"111"<<std::endl;

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
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descOther, layout, dtype, dimNbOther, dimSizeOther.data()));
    const cnnlTensorDescriptor_t input_descs[2] = {descInput, descOther};
    // std::vector<impl::camb::DiopiTensor<diopiTensorHandle_t>> inputs = {trInput, trInput2};
    const void* inputs[2] = {trInput.data(),trOther.data()};
    // inputs.push_back(trInput.data());
    // inputs.push_back(trInput2.data());
    // std::cout<<222<<std::endl;
    // std::vector<void*> inputs = {trInput, trInput2};
    uint32_t input_num = 2;
    // std::cout<<333<<std::endl;
    DIOPI_CALLCNNL(cnnlAddN(handle, input_descs, inputs, input_num, descInput, trOutput.data()));
    std::cout<<"success"<<std::endl;
    cnrtQueueSync(stream);
    return diopiSuccess;


}


}  // extern "C"
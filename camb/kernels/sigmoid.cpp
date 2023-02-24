#include "../helper.hpp"

namespace {
diopiError_t InitTensorDesc(diopiConstTensorHandle_t tensor_handle, cnnlTensorDescriptor_t &desc){
    auto tensor = impl::camb::makeTensor(tensor_handle);
    std::vector<int> src_shape = tensor.shape();
    std::vector<int> shape;
    int dimNb = tensor.shape_len();

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
    InitTensorDesc(tensor_handle, desc);
    auto tensor = impl::camb::makeTensor(tensor_handle);
    ptr = tensor.data();
    return diopiSuccess;
}

diopiError_t InitTensor(diopiConstTensorHandle_t tensor_handle, cnnlTensorDescriptor_t &desc,const void* &ptr){
    InitTensorDesc(tensor_handle, desc);
    auto tensor = impl::camb::makeTensor(tensor_handle);
    ptr = tensor.data();
    return diopiSuccess;
}
}

extern "C" diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input){
    HandleGuard handle_guard;
    auto stream  = impl::camb::getStream(ctx);
    cnnlHandle_t handle = handle_guard.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> CnnlActivation;
    cnnlActivationDescriptor_t activation_desc = CnnlActivation.get();
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v4(activation_desc, 
                                    CNNL_ACTIVATION_RELU, 
                                    CNNL_ACTIVATION_HIGH_PRECISION, 
                                    CNNL_NOT_PROPAGATE_NAN, 
                                    0.0, 
                                    0, 
                                    0.0, 
                                    0.0));

    TensorDescGuard x_desc_guard;
    cnnlTensorDescriptor_t x_desc = x_desc_guard.get();
    void* x_ptr;
    DIOPI_CALL(InitTensor(input, x_desc, x_ptr));
    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc, NULL, x_desc, x_ptr, NULL, x_desc, x_ptr));
}
extern "C" diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input){
    HandleGuard handle_guard;
    auto stream  = impl::camb::getStream(ctx);
    cnnlHandle_t handle = handle_guard.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activate_guard;
    cnnlActivationDescriptor_t activation_desc = activate_guard.get();
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v4(activation_desc, 
                                    CNNL_ACTIVATION_SIGMOID, 
                                    CNNL_ACTIVATION_HIGH_PRECISION, 
                                    CNNL_NOT_PROPAGATE_NAN, 
                                    0.0, 
                                    0, 
                                    0.0, 
                                    0.0));
    TensorDescGuard x_desc_guard, y_desc_guard;
    cnnlTensorDescriptor_t x_desc = x_desc_guard.get();
    cnnlTensorDescriptor_t y_desc = y_desc_guard.get();
    const void* x_ptr;
    void* y_ptr;
    DIOPI_CALL(InitTensor(input, x_desc, x_ptr));
    DIOPI_CALL(InitTensor(out, y_desc, y_ptr));
    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc, NULL, x_desc, x_ptr, NULL, y_desc, y_ptr));
}
#include <diopi/functions.h>
#include <cnnl.h>
#include<iostream>

#include <cstdio>
#include <vector>

#include "helper.hpp"

#define DIOPI_CALLCNNL(Expr) { \
        ::cnnlStatus_t ret = Expr; \
        if (ret != ::CNNL_STATUS_SUCCESS) { \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%d",           \
                    ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__);               \
            return diopiErrorOccurred;                                                 \
        }}\

#define DIOPI_CHECKCNNL(Expr) { \
        ::cnnlStatus_t ret = Expr; \
        if (ret != ::CNNL_STATUS_SUCCESS) { \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%d",           \
                    ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__);               \
        }}\

template<typename T, ::cnnlStatus_t(*fnCreate)(T*), ::cnnlStatus_t(*fnDestroy)(T)>
class CnnlResourceGuard final {
public:
    CnnlResourceGuard() {
        DIOPI_CHECKCNNL(fnCreate(&resource_));
    }

    ~CnnlResourceGuard() {
        DIOPI_CHECKCNNL(fnDestroy(resource_));
    }

    T& get() {
        return resource_;
    }

protected:
    T resource_ {0};
};

static diopiError_t convertType(cnnlDataType_t *cnnlType, diopiDtype_t type) {
    switch (type) {
    case diopi_dtype_int8:
        *cnnlType = CNNL_DTYPE_INT8;
        break;
    case diopi_dtype_uint8:
        *cnnlType = CNNL_DTYPE_UINT8;
        break;
    case diopi_dtype_int32:
        *cnnlType = CNNL_DTYPE_INT32;
        break;
    case diopi_dtype_uint32:
        *cnnlType = CNNL_DTYPE_UINT32;
        break;
    case diopi_dtype_float16:
        *cnnlType = CNNL_DTYPE_HALF;
        break;
    case diopi_dtype_float32:
        *cnnlType = CNNL_DTYPE_FLOAT;
        break;
    case diopi_dtype_int16:
        *cnnlType = CNNL_DTYPE_INT16;
        break;
    case diopi_dtype_uint16:
        *cnnlType = CNNL_DTYPE_UINT16;
        break;
    case diopi_dtype_bool:
        *cnnlType = CNNL_DTYPE_BOOL;
        break;
    case diopi_dtype_int64:
        *cnnlType = CNNL_DTYPE_INT64;
        break;
    default:
        impl::camb::set_last_error_string("unkown diopitype error %d at %s:%s", type, __FILE__, __LINE__);
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto stream  = impl::camb::getStream(ctx);
    auto trInput = impl::camb::makeTensor(input);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDesc;
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));
    cnnlTensorDescriptor_t desc = CnnlDesc.get();

    diopiSize_t shape = trInput.shape();
    int dimNb = shape.len;
    std::vector<int> dimStrides(dimNb, 1);
    std::vector<int> dimSize(dimNb);
    diopiSize_t stride = trInput.stride();

    if (dimNb == 0) {
        dimNb = 1;
        dimSize.push_back(1);
        dimStrides.push_back(1);
    } else {
        for (int i = 0; i < dimNb; ++i) {
            dimSize[i] = shape.data[i];
        }
        if (dimNb > 0) {
            for (int i = 0; i < dimNb; ++i) {
                dimStrides[i] = stride.data[i];
            }
        }
    }

    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }

    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(desc, layout, dtype, dimNb,
        dimSize.data(), dimStrides.data()));
    DIOPI_CALLCNNL(cnnlFill(handle, val, desc, trInput.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha){

    auto stream  = impl::camb::getStream(ctx);
    // diopiTensorHandle_t tmp = (diopiTensorHandle_t)input;
    // auto trInput = impl::camb::makeTensor(tmp);
    // tmp = (diopiTensorHandle_t)other;
    // auto trOther = impl::camb::makeTensor(tmp);
    auto trInput = impl::camb::makeTensor(input);
    auto trOther = impl::camb::makeTensor(other);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDesc;
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));
    cnnlTensorDescriptor_t desc = CnnlDesc.get();

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

    double val;
    if (alpha->stype <= 7) {
        val = alpha->ival;
    } else {
        val = alpha->fval;
    }

    void *Alpha, *Beta;
    std::cout<<"alpha: "<<val<<std::endl;
    if (dtype >=2 && dtype <=7) {
        int32_t *tmp = new int32_t;
        *tmp = (int32_t)val;
        Alpha = tmp;
        *tmp = 0;
        Beta = tmp;
    }
    else{
        double *tmp = new double;
        *tmp = (double)val;
        Alpha = tmp;
        *tmp = 0.0;
        Beta = tmp;
    }


    std::cout<<"other: "<<trOther.data()<<std::endl;
    // std::cout<<"alpha: "<<(*alpha)<<std::endl;
    std::cout<<"input: "<<trInput.data()<<std::endl;

    // std::cout<<"Alpha: "<<*Alpha<<std::endl;
    // std::cout<<"Beta: "<<*Beta<<std::endl;

    // auto output = impl::camb::makeTensor(out);
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(desc, layout, dtype, dimNb, dimSize.data()));
    std::cout<<"000"<<std::endl;
    DIOPI_CALLCNNL(cnnlTransform(handle, Alpha, desc, trOther.data(), Beta, out));

    std::cout<<"111"<<std::endl;
    auto trInput2 = impl::camb::makeTensor(out);
    // auto trInput2 = trOther;
    std::cout<<"output: "<<trInput2.data()<<std::endl;
    std::vector<cnnlTensorDescriptor_t> input_descs = {desc, desc};
    // std::vector<impl::camb::DiopiTensor<diopiTensorHandle_t>> inputs = {&trInput, &trInput2};
    std::vector<void*> inputs = {&trInput, &trInput2};
    uint32_t input_num = 2;
    DIOPI_CALLCNNL(cnnlAddN(handle, input_descs.data(), inputs.data(), input_num, desc, out))

    return diopiSuccess;

}

DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);


}  // extern "C"

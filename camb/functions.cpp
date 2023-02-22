#include <diopi/functions.h>
#include <cnnl.h>

#include <cstdio>
#include <vector>

#include "helper.hpp"
#include <iostream>

#define DIOPI_CALLCNNL(Expr) { \
        ::cnnlStatus_t ret = Expr; \
        if (ret != ::CNNL_STATUS_SUCCESS) { \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s",           \
                    ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__);               \
            return diopiErrorOccurred;                                                 \
        }}\

#define DIOPI_CHECKCNNL(Expr) { \
        ::cnnlStatus_t ret = Expr; \
        if (ret != ::CNNL_STATUS_SUCCESS) { \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s",           \
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

DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    std::cout << "--------------------------------------------------" << std::endl;
    auto stream  = impl::camb::getStream(ctx);

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

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> InputCnnlDesc;
    cnnlTensorDescriptor_t input_desc = InputCnnlDesc.get();
    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> OtherCnnlDesc;
    cnnlTensorDescriptor_t other_desc = OtherCnnlDesc.get();
    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> OutCnnlDesc;
    cnnlTensorDescriptor_t out_desc = OutCnnlDesc.get();

   
    diopiSize_t input_shape = trInput.shape();
    int input_dimNb = input_shape.len;
    std::vector<int> input_dimSize(input_dimNb);
    std::vector<int> input_dimStrides(input_dimNb, 1);
    diopiSize_t input_stride = trInput.stride();

    std::cout << "the input dimNb = " << input_dimNb << std::endl;
    if (input_dimNb == 0) {
        input_dimNb = 1;
        input_dimSize.push_back(1);
        input_dimStrides.push_back(1);
    } else {
        std::cout << "the input_shape is: " << std::endl;
        std::cout << "[";
        for (int i = 0; i < input_dimNb; ++i) {
            input_dimSize[i] = input_shape.data[i];
            std::cout << input_dimSize[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "the input_stride is: " << std::endl;
        std::cout << "[";
        if (input_dimNb > 0) {
            for (int i = 0; i < input_dimNb; ++i) {
                input_dimStrides[i] = input_stride.data[i];
                std::cout << input_dimStrides[i] << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    std::cout << "---------------" << std::endl;

    diopiSize_t other_shape = trOther.shape();
    int other_dimNb = other_shape.len;
    std::vector<int> other_dimSize(other_dimNb);
    std::vector<int> other_dimStrides(other_dimNb, 1);
    diopiSize_t other_stride = trOther.stride();


    std::cout << "the other dimNb = " << other_dimNb << std::endl;
    if (other_dimNb == 0) {
        other_dimNb = 1;
        other_dimSize.push_back(1);
        other_dimStrides.push_back(1);
    } else {
        std::cout << "the other_shape is: " << std::endl;
        std::cout << "[";
        for (int i = 0; i < other_dimNb; ++i) {
            other_dimSize[i] = other_shape.data[i];
            std::cout << other_dimSize[i] << ", ";

        }
        std::cout << "]" << std::endl;

        std::cout << "the other_stride is: " << std::endl;
        std::cout << "[";
        if (other_dimNb > 0) {
            for (int i = 0; i < other_dimNb; ++i) {
                other_dimStrides[i] = other_stride.data[i];
                std::cout << other_dimStrides[i] << ", ";

            }
            std::cout << "]" << std::endl;
        }
    }
    std::cout << "---------------" << std::endl;

    diopiSize_t out_shape = trOut.shape();
    int out_dimNb = out_shape.len;
    std::vector<int> out_dimSize(out_dimNb);
    std::vector<int> out_dimStrides(out_dimNb, 1);
    diopiSize_t out_stride = trOut.stride();

    
    std::cout << "the out dimNb = " << out_dimNb << std::endl;
    if (out_dimNb == 0) { 
        out_dimNb = 1;
        out_dimSize.push_back(1);
        out_dimStrides.push_back(1);
    } else {
        std::cout << "the out_shape is: " << std::endl;
        std::cout << "[";
        for (int i = 0; i < out_dimNb; ++i) {
            out_dimSize[i] = out_shape.data[i];
            std::cout << out_dimSize[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "the out_stride is: " << std::endl;
        std::cout << "[";
        if (out_dimNb > 0) {
            for (int i = 0; i < out_dimNb; ++i) {
                out_dimStrides[i] = out_stride.data[i];
                std::cout << out_dimStrides[i] << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
   
    cnnlSetTensorDescriptorEx(input_desc, input_layout, input_dtype, input_dimNb, input_dimSize.data(), input_dimStrides.data());
    cnnlSetTensorDescriptorEx(other_desc, other_layout, other_dtype, other_dimNb, other_dimSize.data(), other_dimStrides.data());
    cnnlSetTensorDescriptorEx(out_desc,   out_layout,   out_dtype,   out_dimNb,   out_dimSize.data(),   out_dimStrides.data());
     
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc, other_desc, out_desc, &workspace_size));
    cnnlLogicOp_t logic_op_and = CNNL_LOGIC_OP_AND;
    std::cout << "the workspace_size is " << workspace_size << std::endl;
    diopiTensorHandle_t* pworkspace;
    diopiRequireBuffer(ctx, pworkspace, workspace_size, diopi_device);
    void* workspace = trOut.data();
    DIOPI_CALLCNNL(cnnlLogicOp(handle, logic_op_and, input_desc, trInput.data(), other_desc, trOther.data(), 
        *pworkspace, workspace_size, out_desc, trOut.data()));
}
}  // extern "C"

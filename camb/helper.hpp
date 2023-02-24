/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef IMPL_CAMB_HELPER_HPP_
#define IMPL_CAMB_HELPER_HPP_

#include <diopi/diopirt.h>
#include <cnnl.h>
#include <utility>

#include <diopi/functions.h>

#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>

#define DIOPI_CALL(Expr) {    \
    diopiError_t ret = Expr;   \
    if (diopiSuccess != ret) { \
        return ret;            \
    }                          \
}                              \

#define DIOPI_CNNLCHECK(Expr) {                                            \
    ::cnnlStatus_t ret = Expr;                                            \
    if (ret != ::CNNL_STATUS_SUCCESS) {                                   \
        std::cout<<__FILE__ <<", "<< __LINE__<<"\n";                      \
        impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s",  \
                ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__);      \
    }}                                                                    \

#define DIOPI_CALLCNNL(Expr) {                                            \
    ::cnnlStatus_t ret = Expr;                                            \
    if (ret != ::CNNL_STATUS_SUCCESS) {                                   \
        impl::camb::set_last_error_string("cnnl error %d : %s at %s:%d",  \
                ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__);      \
        return diopiErrorOccurred;                                        \
    }}                                                                    \

namespace impl {

namespace camb {

template<typename TensorType>
struct DataType;

template<>
struct DataType<diopiTensorHandle_t> {
    using type = void*;

    static void* data(diopiTensorHandle_t& tensor) {
        void *data;
        diopiGetTensorData(&tensor, &data);
        return data;
    }
};

template<>
struct DataType<diopiConstTensorHandle_t> {
    using type = const void*;
    static const void* data(diopiConstTensorHandle_t& tensor) {
        const void *data;
        diopiGetTensorDataConst(&tensor, &data);
        return data;
    }
};

template<typename TensorType>
class DiopiTensor final {
public:
    explicit DiopiTensor(TensorType& tensor) : tensor_(tensor) {}

    diopiDevice_t device() const {
        diopiDevice_t device;
        diopiGetTensorDevice(tensor_, &device);
        return device;
    }
    diopiDtype_t dtype() const {
        diopiDtype_t dtype;
        diopiGetTensorDtype(tensor_, &dtype);
        return dtype;
    }

    std::vector<int> shape() {
        diopiGetTensorShape(tensor_, &shape_);
        std::vector<int> shape{shape_.data, shape_.data + shape_.len};
        return shape;
    }

    int64_t shape_len(){
        diopiGetTensorShape(tensor_, &shape_);
        int64_t len = shape_.len;
        return len;
    }

    std::vector<int> stride() {
        diopiGetTensorStride(tensor_, &stride_);
        std::vector<int> shape{stride_.data, stride_.data + stride_.len};
        return shape;
    }

    int64_t stride_len(){
        diopiGetTensorStride(tensor_, &stride_);
        int64_t len = stride_.len;
        return len;
    }

    int64_t numel() const {
        int64_t numel;
        diopiGetTensorNumel(tensor_, &numel);
        return numel;
    }
    int64_t elemsize() const {
        int64_t elemsize;
        diopiGetTensorElemSize(tensor_, &elemsize);
        return elemsize;
    }

    typename DataType<TensorType>::type data() {
        return DataType<TensorType>::data(tensor_);
    }

protected:
    TensorType tensor_;
    diopiSize_t shape_;
    diopiSize_t stride_;
};

template<typename TensorType>
auto makeTensor(TensorType& tensor) -> DiopiTensor<TensorType> {
    return DiopiTensor<TensorType>(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresTensor(
        diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return makeTensor(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresBuffer(
        diopiContextHandle_t ctx, int64_t num_bytes) {
    diopiTensorHandle_t tensor;
    diopiRequireBuffer(ctx, &tensor, num_bytes, diopi_device);
    return makeTensor(tensor);
}

inline cnrtQueue_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    return static_cast<cnrtQueue_t>(stream_handle);
}

void _set_last_error_string(const char *err);

template<typename...Types>
void set_last_error_string(const char* szFmt, Types&&...args) {
    char szBuf[4096] = {0};
    sprintf(szBuf, szFmt, std::forward<Types>(args)...);
    _set_last_error_string(szBuf);
}

}  // namespace camb

}  // namespace impl

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
    case diopi_dtype_float64:
        *cnnlType = CNNL_DTYPE_DOUBLE;
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

template<typename T, ::cnnlStatus_t(*fnCreate)(T*), ::cnnlStatus_t(*fnDestroy)(T)>
class CnnlResourceGuard final {
public:
    CnnlResourceGuard() {
        DIOPI_CNNLCHECK(fnCreate(&resource_));
    }

    ~CnnlResourceGuard() {
        DIOPI_CNNLCHECK(fnDestroy(resource_));
    }

    T& get() {
        return resource_;
    }

protected:
    T resource_ {0};
};

typedef CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> HandleGuard;
typedef CnnlResourceGuard<cnnlTensorDescriptor_t,cnnlCreateTensorDescriptor,cnnlDestroyTensorDescriptor> TensorDescGuard;

#endif  // IMPL_CAMB_HELPER_HPP_

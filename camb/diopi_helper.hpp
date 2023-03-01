/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef IMPL_CAMB_DIOPI_HELPER_HPP_
#define IMPL_CAMB_DIOPI_HELPER_HPP_

#include <cnrt.h>
#include <diopi/diopirt.h>

#include <cstdio>
#include <utility>

#include "error.hpp"

#define DIOPI_CHECK(cond, str)                                                         \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            impl::camb::set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                 \
        }                                                                              \
    } while (false);

#define DIOPI_CHECK_NULLPTR(variable)     \
    do {                                  \
        if (variable == nullptr) {                                                                 \
            throw std::runtime_error("The variable `" #variable "` is not defined");     \
        }                                                                 \
    } while (false);

#define DIOPI_CALL(Expr)           \
    do {                           \
        diopiError_t ret = Expr;   \
        if (diopiSuccess != ret) { \
            return ret;            \
        }                          \
    } while (false);

namespace impl {

namespace camb {

template <typename TensorType>
struct DataType;

template <>
struct DataType<diopiTensorHandle_t> {
    using type = void*;

    static void* data(diopiTensorHandle_t& tensor) {
        void* data;
        diopiGetTensorData(&tensor, &data);
        return data;
    }
};

template <>
struct DataType<diopiConstTensorHandle_t> {
    using type = const void*;
    static const void* data(diopiConstTensorHandle_t& tensor) {
        const void* data;
        diopiGetTensorDataConst(&tensor, &data);
        return data;
    }
};

template <typename TensorType>
class DiopiTensor final {
public:
    explicit DiopiTensor(TensorType& tensor) : tensor_(tensor) {}

    diopiDevice_t device() const {
        DIOPI_CHECK_NULLPTR(tensor_);
        diopiDevice_t device;
        diopiGetTensorDevice(tensor_, &device);
        return device;
    }
    diopiDtype_t dtype() const {
        DIOPI_CHECK_NULLPTR(tensor_);
        diopiDtype_t dtype;
        diopiGetTensorDtype(tensor_, &dtype);
        return dtype;
    }

    const diopiSize_t& shape() {
        DIOPI_CHECK_NULLPTR(tensor_);
        diopiGetTensorShape(tensor_, &shape_);
        return shape_;
    }
    const diopiSize_t& stride() {
        DIOPI_CHECK_NULLPTR(tensor_);
        diopiGetTensorStride(tensor_, &stride_);
        return stride_;
    }

    int64_t numel() const {
        DIOPI_CHECK_NULLPTR(tensor_);
        int64_t numel;
        diopiGetTensorNumel(tensor_, &numel);
        return numel;
    }
    int64_t elemsize() const {
        DIOPI_CHECK_NULLPTR(tensor_);
        int64_t elemsize;
        diopiGetTensorElemSize(tensor_, &elemsize);
        return elemsize;
    }
    int64_t dim() {
        return this->shape().len;
    }
    bool defined() const {
        return tensor_ != nullptr;
    }

    typename DataType<TensorType>::type data() { return DataType<TensorType>::data(tensor_); }

protected:
    TensorType tensor_;

    diopiSize_t shape_;
    diopiSize_t stride_;
};

template <typename TensorType>
inline auto makeTensor(TensorType& tensor) -> DiopiTensor<TensorType> {
    return DiopiTensor<TensorType>(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return makeTensor(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresBuffer(diopiContextHandle_t ctx, int64_t num_bytes) {
    diopiTensorHandle_t tensor;
    diopiRequireBuffer(ctx, &tensor, num_bytes, diopi_device);
    return makeTensor(tensor);
}

inline cnrtQueue_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    return static_cast<cnrtQueue_t>(stream_handle);
}

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_DIOPI_HELPER_HPP_

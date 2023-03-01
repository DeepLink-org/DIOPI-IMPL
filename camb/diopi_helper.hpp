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
#include <vector>

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
    explicit DiopiTensor(TensorType& tensor) : tensor_(tensor) {

        diopiSize_t diopiShape;
        diopiSize_t diopiStride;
        diopiGetTensorShape(tensor_, &diopiShape);
        std::vector<int32_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
        diopiGetTensorStride(tensor_, &diopiStride);
        std::vector<int32_t> strideTmp(diopiStride.data, diopiStride.data + diopiStride.len);
        shape_ = std::move(shapeTmp);
        stride_ = std::move(strideTmp);
    }

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

    const std::vector<int32_t>& shape() {
        return shape_;
    }
    const std::vector<int32_t>& stride() {
        return stride_;
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

    typename DataType<TensorType>::type data() { return DataType<TensorType>::data(tensor_); }

protected:
    TensorType tensor_;
    std::vector<int32_t> shape_;
    std::vector<int32_t> stride_;
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

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
#include <vector>
#include <iostream>


#define DIOPI_CALL(Expr) {                                                              \
    diopiError_t ret = Expr;                                                            \
    if (diopiSuccess != ret) {                                                          \
        return ret;                                                                     \
    }}\


namespace impl {

namespace camb {

enum class MemoryFormat : size_t {
    Contiguous      = 0,
    ChannelsLast    = 1,
    ChannelsLast3d  = 2,
    Preserve        = 3
};

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
struct DataType<const diopiTensorHandle_t> {
    static const void* data(diopiConstTensorHandle_t tensor) {
        const void *data;
        diopiGetTensorDataConst(&tensor, &data);
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

    // TODO: 询问能否获取context（需要改header），这样contiguous就不用传ctx了
    // diopiContextHandle_t context() const {
    //     diopiContextHandle_t context;
    //     _diopiTensorGetCtxHandle(tensor_, &context);
    //     return context;
    // }
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

    const diopiSize_t& shape() {
        diopiGetTensorShape(tensor_, &shape_);
        return shape_;
    }
    const diopiSize_t& stride() {
        diopiGetTensorStride(tensor_, &stride_);
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
    int64_t dim() {
        return this->shape().len;
    }
    bool defined() const {
        return this->numel() != 0;
    }

    DiopiTensor unsqueeze(int dim) {
        // TODO
        return *this;
    }
    DiopiTensor squeeze(int dim) {
        // TODO
        return *this;
    }
    DiopiTensor<diopiTensorHandle_t> contiguous(diopiContextHandle_t ctx, impl::camb::MemoryFormat format) {
        /* Returns a new Tensor in new memory format, without data copy */
        size_t dim = this->dim();
        std::vector<int64_t> strides(dim);
        int64_t stride = 1;
        auto shapes = this->shape().data;
        if (format == impl::camb::MemoryFormat::Contiguous) {
            for (size_t i = dim; i > 0; --i) {
                strides[i - 1] = stride;
                if (shapes[i - 1] == 0) continue;
                if (shapes[i - 1] == -1) stride = -1;
                if (stride != -1) stride *= shapes[i - 1];
            }
        } else if (format == impl::camb::MemoryFormat::ChannelsLast) {
            for (auto k : {1, 3, 2, 0}) {
                strides[k] = stride;
                if (shapes[k] == 0) continue;
                if (shapes[k] == -1) stride = -1;
                if (stride != -1) stride *= shapes[k];
            }
        } 
        diopiSize_t diopi_stride(strides.data(), static_cast<int64_t>(strides.size()));
        diopiTensorHandle_t tensor;
        diopiRequireTensor(ctx, &tensor, &this->shape(), &diopi_stride, this->dtype(), this->device());
        return DiopiTensor<diopiTensorHandle_t>(tensor);
    }
    void print_str() {
        int dim = this->dim();
        std::cout << "DiopiTensor: dim " << dim << ", shape: [";
        for (size_t i = 0; i < dim; i++)
        {
            std::cout << this->shape().data[i] << ", ";
        }
        std::cout << "], stride: [";
        for (size_t i = 0; i < dim; i++)
        {
            std::cout << this->stride().data[i] << ", ";
        }
        std::cout << "] pointer address: " << this->data() << std::endl;
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

#endif  // IMPL_CAMB_HELPER_HPP_

#include <string.h>
#include <iostream>
#include <memory>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiMaskedFill(
    diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = makeTensor(input);
    auto mask_tensor = makeTensor(mask);
    auto value_tensor = makeTensor(value);
    auto out_tensor = makeTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mask_desc(mask_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc value_desc;
    if (value_tensor.shape().size() > 0) {
        DIOPI_CALL(value_desc.set(value_tensor, CNNL_LAYOUT_ARRAY));
    } else {
        std::vector<int> value_dims = {1};
        DIOPI_CALL(value_desc.set(value_tensor, CNNL_LAYOUT_ARRAY, value_dims));
    }

    DiopiTensorT value_cast_tensor;
    CnnlTensorDesc value_cast_desc;
    DiopiTensorT mask_cast_tensor;
    CnnlTensorDesc mask_cast_desc;

    bool value_cast = false;
    bool mask_cast = false;
    if (input_tensor.dtype() != value_tensor.dtype()) {
        value_cast = true;
        value_cast_tensor = dataTypeCast(ctx, value_tensor, input_tensor.dtype());
        value_cast_desc.set(value_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    if ((mask_tensor.dtype() != diopi_dtype_int8) && (mask_tensor.dtype() != diopi_dtype_uint8) && (mask_tensor.dtype() != diopi_dtype_bool)) {
        mask_cast = true;
        mask_cast_tensor = dataTypeCast(ctx, mask_tensor, diopi_dtype_int8);
        mask_cast_desc.set(mask_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle,
                                              CNNL_MASKED_FILL,
                                              input_desc.get(),
                                              mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                              value_cast ? value_cast_desc.get() : value_desc.get(),
                                              out_desc.get(),
                                              &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 CNNL_MASKED_FILL,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                 mask_cast ? mask_cast_tensor.data() : mask_tensor.data(),
                                 value_cast ? value_cast_desc.get() : value_desc.get(),
                                 value_cast ? value_cast_tensor.data() : value_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 out_desc.get(),
                                 out_tensor.data(),
                                 nullptr));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = makeTensor(input);
    auto mask_tensor = makeTensor(mask);
    auto value_tensor = makeTensor(value);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mask_desc(mask_tensor, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc value_desc;
    if (value_tensor.shape().size() > 0) {
        DIOPI_CALL(value_desc.set(value_tensor, CNNL_LAYOUT_ARRAY));
    } else {
        std::vector<int> value_dims = {1};
        DIOPI_CALL(value_desc.set(value_tensor, CNNL_LAYOUT_ARRAY, value_dims));
    }

    DiopiTensorT value_cast_tensor;
    CnnlTensorDesc value_cast_desc;
    DiopiTensorT mask_cast_tensor;
    CnnlTensorDesc mask_cast_desc;

    bool value_cast = false;
    bool mask_cast = false;
    if (input_tensor.dtype() != value_tensor.dtype()) {
        value_cast = true;
        value_cast_tensor = dataTypeCast(ctx, value_tensor, input_tensor.dtype());
        value_cast_desc.set(value_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    if ((mask_tensor.dtype() != diopi_dtype_int8) && (mask_tensor.dtype() != diopi_dtype_uint8) && (mask_tensor.dtype() != diopi_dtype_bool)) {
        mask_cast = true;
        mask_cast_tensor = dataTypeCast(ctx, mask_tensor, diopi_dtype_int8);
        mask_cast_desc.set(mask_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle,
                                              CNNL_MASKED_FILL,
                                              input_desc.get(),
                                              mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                              value_cast ? value_cast_desc.get() : value_desc.get(),
                                              input_desc.get(),
                                              &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 CNNL_MASKED_FILL,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                 mask_cast ? mask_cast_tensor.data() : mask_tensor.data(),
                                 value_cast ? value_cast_desc.get() : value_desc.get(),
                                 value_cast ? value_cast_tensor.data() : value_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 nullptr));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedFillScalar(
    diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = makeTensor(input);
    auto mask_tensor = makeTensor(mask);
    auto out_tensor = makeTensor(out);

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, input_tensor.dtype()));

    diopiDtype_t value_dtype;
    std::unique_ptr<void, void (*)(void*)> value_(malloc(4), free);
    if (CnnlDataType::isInteger(dtype)) {
        value_dtype = diopi_dtype_int32;
        if (value->stype <= 7) {
            *reinterpret_cast<int32_t*>(value_.get()) = static_cast<int32_t>(value->ival);
        } else {
            *reinterpret_cast<int32_t*>(value_.get()) = static_cast<int32_t>(static_cast<float>(value->fval));
        }
    } else {
        value_dtype = diopi_dtype_float32;
        if (value->stype <= 7) {
            *reinterpret_cast<float*>(value_.get()) = static_cast<float>(static_cast<int32_t>(value->ival));
        } else {
            *reinterpret_cast<float*>(value_.get()) = static_cast<float>(value->fval);
        }
    }

    diopiTensorHandle_t value_t;
    std::vector<int64_t> shape_{1};
    diopiSize_t value_shape = vec2diopiSize_t(shape_);
    DIOPI_CALL(diopiRequireTensor(ctx, &value_t, &value_shape, nullptr, value_dtype, diopi_device));
    auto value_tensor = makeTensor(value_t);
    CnnlTensorDesc value_desc(value_tensor, CNNL_LAYOUT_ARRAY);
    auto ret = cnrtMemcpy(value_tensor.data(), value_.get(), 4, cnrtMemcpyHostToDev);
    if (ret != cnrtSuccess) {
        set_last_error_string("%s%d", "cnrt memcpy error, ret = ", ret);
        return diopiErrorOccurred;
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mask_desc(mask_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    DiopiTensorT value_cast_tensor;
    CnnlTensorDesc value_cast_desc;
    DiopiTensorT mask_cast_tensor;
    CnnlTensorDesc mask_cast_desc;

    bool value_cast = false;
    bool mask_cast = false;
    if (input_tensor.dtype() != value_tensor.dtype()) {
        value_cast = true;
        value_cast_tensor = dataTypeCast(ctx, value_tensor, input_tensor.dtype());
        value_cast_desc.set(value_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    if ((mask_tensor.dtype() != diopi_dtype_int8) && (mask_tensor.dtype() != diopi_dtype_uint8) && (mask_tensor.dtype() != diopi_dtype_bool)) {
        mask_cast = true;
        mask_cast_tensor = dataTypeCast(ctx, mask_tensor, diopi_dtype_int8);
        mask_cast_desc.set(mask_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle,
                                              CNNL_MASKED_FILL,
                                              input_desc.get(),
                                              mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                              value_cast ? value_cast_desc.get() : value_desc.get(),
                                              out_desc.get(),
                                              &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 CNNL_MASKED_FILL,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                 mask_cast ? mask_cast_tensor.data() : mask_tensor.data(),
                                 value_cast ? value_cast_desc.get() : value_desc.get(),
                                 value_cast ? value_cast_tensor.data() : value_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 out_desc.get(),
                                 out_tensor.data(),
                                 nullptr));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t input,
                                                diopiConstTensorHandle_t mask,
                                                const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = makeTensor(input);
    auto mask_tensor = makeTensor(mask);

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, input_tensor.dtype()));

    diopiDtype_t value_dtype;
    std::unique_ptr<void, void (*)(void*)> value_(malloc(4), free);
    if (CnnlDataType::isInteger(dtype)) {
        value_dtype = diopi_dtype_int32;
        if (value->stype <= 7) {
            *reinterpret_cast<int32_t*>(value_.get()) = static_cast<int32_t>(value->ival);
        } else {
            *reinterpret_cast<int32_t*>(value_.get()) = static_cast<int32_t>(static_cast<float>(value->fval));
        }
    } else {
        value_dtype = diopi_dtype_float32;
        if (value->stype <= 7) {
            *reinterpret_cast<float*>(value_.get()) = static_cast<float>(static_cast<int32_t>(value->ival));
        } else {
            *reinterpret_cast<float*>(value_.get()) = static_cast<float>(value->fval);
        }
    }

    diopiTensorHandle_t value_t;
    std::vector<int64_t> shape_{1};
    diopiSize_t value_shape = vec2diopiSize_t(shape_);
    DIOPI_CALL(diopiRequireTensor(ctx, &value_t, &value_shape, nullptr, value_dtype, diopi_device));
    auto value_tensor = makeTensor(value_t);
    CnnlTensorDesc value_desc(value_tensor, CNNL_LAYOUT_ARRAY);
    auto ret = cnrtMemcpy(value_tensor.data(), value_.get(), 4, cnrtMemcpyHostToDev);
    if (ret != cnrtSuccess) {
        set_last_error_string("%s%d", "cnrt memcpy error, ret = ", ret);
        return diopiErrorOccurred;
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mask_desc(mask_tensor, CNNL_LAYOUT_ARRAY);

    DiopiTensorT value_cast_tensor;
    CnnlTensorDesc value_cast_desc;
    DiopiTensorT mask_cast_tensor;
    CnnlTensorDesc mask_cast_desc;

    bool value_cast = false;
    bool mask_cast = false;
    if (input_tensor.dtype() != value_tensor.dtype()) {
        value_cast = true;
        value_cast_tensor = dataTypeCast(ctx, value_tensor, input_tensor.dtype());
        value_cast_desc.set(value_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    if ((mask_tensor.dtype() != diopi_dtype_int8) && (mask_tensor.dtype() != diopi_dtype_uint8) && (mask_tensor.dtype() != diopi_dtype_bool)) {
        mask_cast = true;
        mask_cast_tensor = dataTypeCast(ctx, mask_tensor, diopi_dtype_int8);
        mask_cast_desc.set(mask_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle,
                                              CNNL_MASKED_FILL,
                                              input_desc.get(),
                                              mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                              value_cast ? value_cast_desc.get() : value_desc.get(),
                                              input_desc.get(),
                                              &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 CNNL_MASKED_FILL,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 mask_cast ? mask_cast_desc.get() : mask_desc.get(),
                                 mask_cast ? mask_cast_tensor.data() : mask_tensor.data(),
                                 value_cast ? value_cast_desc.get() : value_desc.get(),
                                 value_cast ? value_cast_tensor.data() : value_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 nullptr));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

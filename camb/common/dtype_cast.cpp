/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cnrt.h>

#include <memory>
#include <set>

#include "common.hpp"

namespace impl {
namespace camb {

#define _MAKE_KEY(a, b) (((static_cast<uint64_t>(a) & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(b) & 0xFFFFFFFF))

constexpr uint64_t Float64Float32 = _MAKE_KEY(diopi_dtype_float64, diopi_dtype_float32);
constexpr uint64_t Float32Float64 = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_float64);
constexpr uint64_t Float32Float16 = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_float16);
constexpr uint64_t Float32Int64 = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_int64);
constexpr uint64_t Float32Int32 = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_int32);
constexpr uint64_t Float32Int16 = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_int16);
constexpr uint64_t Float32Int8 = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_int8);
constexpr uint64_t Float32Uint8 = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_uint8);
constexpr uint64_t Float32Bool = _MAKE_KEY(diopi_dtype_float32, diopi_dtype_bool);
constexpr uint64_t Float16Float32 = _MAKE_KEY(diopi_dtype_float16, diopi_dtype_float32);
constexpr uint64_t Float16Int64 = _MAKE_KEY(diopi_dtype_float16, diopi_dtype_int64);
constexpr uint64_t Float16Int32 = _MAKE_KEY(diopi_dtype_float16, diopi_dtype_int32);
constexpr uint64_t Float16Int16 = _MAKE_KEY(diopi_dtype_float16, diopi_dtype_int16);
constexpr uint64_t Float16Int8 = _MAKE_KEY(diopi_dtype_float16, diopi_dtype_int8);
constexpr uint64_t Float16Uint8 = _MAKE_KEY(diopi_dtype_float16, diopi_dtype_uint8);
constexpr uint64_t Float16Bool = _MAKE_KEY(diopi_dtype_float16, diopi_dtype_bool);
constexpr uint64_t Int32Float32 = _MAKE_KEY(diopi_dtype_int32, diopi_dtype_float32);
constexpr uint64_t Int32Float16 = _MAKE_KEY(diopi_dtype_int32, diopi_dtype_float16);
constexpr uint64_t Int32Bool = _MAKE_KEY(diopi_dtype_int32, diopi_dtype_bool);
constexpr uint64_t Int32Int8 = _MAKE_KEY(diopi_dtype_int32, diopi_dtype_int8);
constexpr uint64_t Int32Int16 = _MAKE_KEY(diopi_dtype_int32, diopi_dtype_int16);
constexpr uint64_t Int32Int64 = _MAKE_KEY(diopi_dtype_int32, diopi_dtype_int64);
constexpr uint64_t Uint32Int64 = _MAKE_KEY(diopi_dtype_uint32, diopi_dtype_int64);
constexpr uint64_t Int16Float32 = _MAKE_KEY(diopi_dtype_int16, diopi_dtype_float32);
constexpr uint64_t Int16Float16 = _MAKE_KEY(diopi_dtype_int16, diopi_dtype_float16);
constexpr uint64_t Int16Int32 = _MAKE_KEY(diopi_dtype_int16, diopi_dtype_int32);
constexpr uint64_t Int8Float32 = _MAKE_KEY(diopi_dtype_int8, diopi_dtype_float32);
constexpr uint64_t Int8Float16 = _MAKE_KEY(diopi_dtype_int8, diopi_dtype_float16);
constexpr uint64_t Int8Int32 = _MAKE_KEY(diopi_dtype_int8, diopi_dtype_int32);
constexpr uint64_t Uint8Float32 = _MAKE_KEY(diopi_dtype_uint8, diopi_dtype_float32);
constexpr uint64_t Uint8Float16 = _MAKE_KEY(diopi_dtype_uint8, diopi_dtype_float16);
constexpr uint64_t Uint8Int32 = _MAKE_KEY(diopi_dtype_uint8, diopi_dtype_int32);
constexpr uint64_t Uint8Int64 = _MAKE_KEY(diopi_dtype_uint8, diopi_dtype_int64);
constexpr uint64_t BoolFloat32 = _MAKE_KEY(diopi_dtype_bool, diopi_dtype_float32);
constexpr uint64_t BoolFloat16 = _MAKE_KEY(diopi_dtype_bool, diopi_dtype_float16);
constexpr uint64_t BoolInt32 = _MAKE_KEY(diopi_dtype_bool, diopi_dtype_int32);
constexpr uint64_t Int64Int32 = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_int32);
constexpr uint64_t Int64Uint32 = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_uint32);
constexpr uint64_t Int64Float16 = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_float16);
constexpr uint64_t Int64Float32 = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_float32);

// special convert (cnnl doesn't support)
constexpr uint64_t BoolInt64 = _MAKE_KEY(diopi_dtype_bool, diopi_dtype_int64);
constexpr uint64_t Int16Int64 = _MAKE_KEY(diopi_dtype_int16, diopi_dtype_int64);
constexpr uint64_t Uint8Bool = _MAKE_KEY(diopi_dtype_uint8, diopi_dtype_bool);
constexpr uint64_t Int16Bool = _MAKE_KEY(diopi_dtype_int16, diopi_dtype_bool);
constexpr uint64_t Int64Bool = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_bool);
constexpr uint64_t Int8Bool = _MAKE_KEY(diopi_dtype_int8, diopi_dtype_bool);
constexpr uint64_t Int8Int64 = _MAKE_KEY(diopi_dtype_int8, diopi_dtype_int64);
constexpr uint64_t Int64Int8 = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_int8);

#define CAST_TYPE(TYPE)   \
    {                     \
        cast_type = TYPE; \
        break;            \
    }

// Cast through middle type
// TODO(waiting for dispatch), after log system is built, warning here
#define CAST_TYPE_THROUGH_INT32(TYPE)                   \
    {                                                   \
        dataTypeCast(ctx, input_tr, diopi_dtype_int32); \
        cast_type = TYPE;                               \
        break;                                          \
    }

diopiError_t dataTypeCastInternal(diopiContextHandle_t ctx, DiopiTensor input_tr, DiopiTensor output_tr) {

    if (input_tr.dtype() == output_tr.dtype()) {
        return diopiSuccess;
    }

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto key = _MAKE_KEY(input_tr.dtype(), output_tr.dtype());
    cnnlCastDataType_t cast_type;
    switch (key) {
        // cast once
        case Float64Float32:
            CAST_TYPE(CNNL_CAST_DOUBLE_TO_FLOAT)
        case Float32Float64:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_DOUBLE)
        case Float32Float16:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_HALF)
        case Float32Int64:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_INT64)
        case Float32Int32:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_INT32)
        case Float32Int16:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_INT16)
        case Float32Int8:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_INT8)
        case Float32Uint8:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_UINT8)
        case Float32Bool:
            CAST_TYPE(CNNL_CAST_FLOAT_TO_BOOL)
        case Float16Float32:
            CAST_TYPE(CNNL_CAST_HALF_TO_FLOAT_INF)
        case Float16Int64:
            CAST_TYPE(CNNL_CAST_HALF_TO_INT64)
        case Float16Int32:
            CAST_TYPE(CNNL_CAST_HALF_TO_INT32)
        case Float16Int16:
            CAST_TYPE(CNNL_CAST_HALF_TO_INT16)
        case Float16Int8:
            CAST_TYPE(CNNL_CAST_HALF_TO_INT8)
        case Float16Uint8:
            CAST_TYPE(CNNL_CAST_HALF_TO_UINT8)
        case Float16Bool:
            CAST_TYPE(CNNL_CAST_HALF_TO_BOOL)
        case Int32Float32:
            CAST_TYPE(CNNL_CAST_INT32_TO_FLOAT)
        case Int32Float16:
            CAST_TYPE(CNNL_CAST_INT32_TO_HALF)
        case Int32Bool:
            CAST_TYPE(CNNL_CAST_INT32_TO_BOOL)
        case Int32Int8:
            CAST_TYPE(CNNL_CAST_INT32_TO_INT8)
        case Int32Int16:
            CAST_TYPE(CNNL_CAST_INT32_TO_INT16)
        case Int32Int64:
            CAST_TYPE(CNNL_CAST_INT32_TO_INT64)
        case Uint32Int64:
            CAST_TYPE(CNNL_CAST_UINT32_TO_INT64)
        case Int16Float32:
            CAST_TYPE(CNNL_CAST_INT16_TO_FLOAT)
        case Int16Float16:
            CAST_TYPE(CNNL_CAST_INT16_TO_HALF)
        case Int16Int32:
            CAST_TYPE(CNNL_CAST_INT16_TO_INT32)
        case Int8Float32:
            CAST_TYPE(CNNL_CAST_INT8_TO_FLOAT)
        case Int8Float16:
            CAST_TYPE(CNNL_CAST_INT8_TO_HALF)
        case Int8Int32:
            CAST_TYPE(CNNL_CAST_INT8_TO_INT32)
        case Uint8Float32:
            CAST_TYPE(CNNL_CAST_UINT8_TO_FLOAT)
        case Uint8Float16:
            CAST_TYPE(CNNL_CAST_UINT8_TO_HALF)
        case Uint8Int32:
            CAST_TYPE(CNNL_CAST_UINT8_TO_INT32)
        case Uint8Int64:
            CAST_TYPE(CNNL_CAST_UINT8_TO_INT64)
        case BoolFloat32:
            CAST_TYPE(CNNL_CAST_BOOL_TO_FLOAT)
        case BoolFloat16:
            CAST_TYPE(CNNL_CAST_BOOL_TO_HALF)
        case BoolInt32:
            CAST_TYPE(CNNL_CAST_BOOL_TO_INT32)
        case Int64Int32:
            CAST_TYPE(CNNL_CAST_INT64_TO_INT32)
        case Int64Uint32:
            CAST_TYPE(CNNL_CAST_INT64_TO_UINT32)
        case Int64Float16:
            CAST_TYPE(CNNL_CAST_INT64_TO_HALF)
        case Int64Float32:
            CAST_TYPE(CNNL_CAST_INT64_TO_FLOAT)

        // cast through middle
        case BoolInt64:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_INT64)
        case Int16Int64:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_INT64)
        case Int8Int64:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_INT64)
        case Uint8Bool:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)
        case Int16Bool:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)
        case Int64Bool:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)
        case Int8Bool:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)
        case Int64Int8:
            CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_INT8)

        default:
            set_last_error_string("Can not cast from %d to %d at %s:%d ", input_tr.dtype(), output_tr.dtype(), __FILE__, __LINE__);
            return diopiDtypeNotSupported;
    }

    CnnlTensorDesc input_desc(input_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tr, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECKCNNL(cnnlCastDataType(handle, input_desc.get(), input_tr.data(), cast_type, output_desc.get(), output_tr.data()));

    return diopiSuccess;
}

diopiError_t dataTypeCast(diopiContextHandle_t& ctx, DiopiTensor& src, diopiDtype_t destDtype) {
    DiopiTensor dest = requiresTensor(ctx, src.shape(), destDtype);
    DIOPI_CALL(dataTypeCastInternal(ctx, src, dest));
    src = dest;
    return diopiSuccess;
}

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src) {
    // check size of dest and src
    DIOPI_CHECK(src.shape() == dest.shape(), "the shapes of src and dest are not equal");

    return dataTypeCastInternal(ctx, src, dest);
}

static diopiError_t choiceDtype(const std::set<diopiDtype_t>& opSupportedDtypes, diopiDtype_t* dtype) {
    if (opSupportedDtypes.find(diopi_dtype_float32) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_float32;
    } else if (opSupportedDtypes.find(diopi_dtype_float16) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_float16;
    } else if (opSupportedDtypes.find(diopi_dtype_int32) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_int32;
    } else if (opSupportedDtypes.find(diopi_dtype_int16) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_int16;
    } else if (opSupportedDtypes.find(diopi_dtype_int8) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_int8;
    } else if (opSupportedDtypes.find(diopi_dtype_bool) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_bool;
    } else {
        set_last_error_string("this operator does not support bool, int8, int16, int32, float16, float32");
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype) {
    // std::multimap<diopiDtype_t, DiopiTensor*> dtypeAndTensorPtrs;
    std::set<diopiDtype_t> dtypeAndTensorPtrs;
    diopiDtype_t targetType = diopi_dtype_float32;
    for (const auto& pTensor : pTensors) {
        dtypeAndTensorPtrs.insert(pTensor->dtype());
    }
    if (dtypeAndTensorPtrs.find(diopi_dtype_float64) != dtypeAndTensorPtrs.end() || dtypeAndTensorPtrs.find(diopi_dtype_float32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float32) == opSupportedDtype.end()) {  // not support float32
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into float32
            targetType = diopi_dtype_float32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_float16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float16) == opSupportedDtype.end()) {  // not support float16
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into float16
            targetType = diopi_dtype_float16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_int32) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int32) == opSupportedDtype.end()) {  // not support int32
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into int32
            targetType = diopi_dtype_int32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int16) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int16) == opSupportedDtype.end()) {  // not support int16
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into int16
            targetType = diopi_dtype_int16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int8) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint8) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int8) == opSupportedDtype.end()) {  // not support int8
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into int8
            targetType = diopi_dtype_int8;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_bool) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_bool) == opSupportedDtype.end()) {  // not support bool
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into bool
            targetType = diopi_dtype_bool;
        }
    } else {
        set_last_error_string("tensor's dtype error, can't be cast");
        return diopiDtypeNotSupported;
    }
    for (const auto& pTensor : pTensors) {
        DIOPI_CALL(dataTypeCast(ctx, *pTensor, targetType));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

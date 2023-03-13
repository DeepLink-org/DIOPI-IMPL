#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

#define _MAKE_KEY(a, b) \
    (((static_cast<uint64_t>(a) & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(b) & 0xFFFFFFFF))

constexpr uint64_t Float64Float32   = _MAKE_KEY(diopiDtype_t::diopi_dtype_float64, diopiDtype_t::diopi_dtype_float32);
constexpr uint64_t Float32Float64   = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_float64);
constexpr uint64_t Float32Float16   = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_float16);
constexpr uint64_t Float32Int64     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_int64);
constexpr uint64_t Float32Int32     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_int32);
constexpr uint64_t Float32Int16     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_int16);
constexpr uint64_t Float32Int8      = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_int8);
constexpr uint64_t Float32Uint8     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_uint8);
constexpr uint64_t Float32Bool      = _MAKE_KEY(diopiDtype_t::diopi_dtype_float32, diopiDtype_t::diopi_dtype_bool);
constexpr uint64_t Float16Float32   = _MAKE_KEY(diopiDtype_t::diopi_dtype_float16, diopiDtype_t::diopi_dtype_float32);
constexpr uint64_t Float16Int64     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float16, diopiDtype_t::diopi_dtype_int64);
constexpr uint64_t Float16Int32     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float16, diopiDtype_t::diopi_dtype_int32);
constexpr uint64_t Float16Int16     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float16, diopiDtype_t::diopi_dtype_int16);
constexpr uint64_t Float16Int8      = _MAKE_KEY(diopiDtype_t::diopi_dtype_float16, diopiDtype_t::diopi_dtype_int8);
constexpr uint64_t Float16Uint8     = _MAKE_KEY(diopiDtype_t::diopi_dtype_float16, diopiDtype_t::diopi_dtype_uint8);
constexpr uint64_t Float16Bool      = _MAKE_KEY(diopiDtype_t::diopi_dtype_float16, diopiDtype_t::diopi_dtype_bool);
constexpr uint64_t Int32Float32     = _MAKE_KEY(diopiDtype_t::diopi_dtype_int32, diopiDtype_t::diopi_dtype_float32);
constexpr uint64_t Int32Float16     = _MAKE_KEY(diopiDtype_t::diopi_dtype_int32, diopiDtype_t::diopi_dtype_float16);
constexpr uint64_t Int32Bool        = _MAKE_KEY(diopiDtype_t::diopi_dtype_int32, diopiDtype_t::diopi_dtype_bool);
constexpr uint64_t Int32Int8        = _MAKE_KEY(diopiDtype_t::diopi_dtype_int32, diopiDtype_t::diopi_dtype_int8);
constexpr uint64_t Int32Int16       = _MAKE_KEY(diopiDtype_t::diopi_dtype_int32, diopiDtype_t::diopi_dtype_int16);
constexpr uint64_t Int32Int64       = _MAKE_KEY(diopiDtype_t::diopi_dtype_int32, diopiDtype_t::diopi_dtype_int64);
constexpr uint64_t Uint32Int64      = _MAKE_KEY(diopiDtype_t::diopi_dtype_uint32, diopiDtype_t::diopi_dtype_int64);
constexpr uint64_t Int16Float32     = _MAKE_KEY(diopiDtype_t::diopi_dtype_int16, diopiDtype_t::diopi_dtype_float32);
constexpr uint64_t Int16Float16     = _MAKE_KEY(diopiDtype_t::diopi_dtype_int16, diopiDtype_t::diopi_dtype_float16);
constexpr uint64_t Int16Int32       = _MAKE_KEY(diopiDtype_t::diopi_dtype_int16, diopiDtype_t::diopi_dtype_int32);
constexpr uint64_t Int8Float32      = _MAKE_KEY(diopiDtype_t::diopi_dtype_int8, diopiDtype_t::diopi_dtype_float32);
constexpr uint64_t Int8Float16      = _MAKE_KEY(diopiDtype_t::diopi_dtype_int8, diopiDtype_t::diopi_dtype_float16);
constexpr uint64_t Int8Int32        = _MAKE_KEY(diopiDtype_t::diopi_dtype_int8, diopiDtype_t::diopi_dtype_int32);
constexpr uint64_t Uint8Float32     = _MAKE_KEY(diopiDtype_t::diopi_dtype_uint8, diopiDtype_t::diopi_dtype_float32);
constexpr uint64_t Uint8Float16     = _MAKE_KEY(diopiDtype_t::diopi_dtype_uint8, diopiDtype_t::diopi_dtype_float16);
constexpr uint64_t Uint8Int32       = _MAKE_KEY(diopiDtype_t::diopi_dtype_uint8, diopiDtype_t::diopi_dtype_int32);
constexpr uint64_t Uint8Int64       = _MAKE_KEY(diopiDtype_t::diopi_dtype_uint8, diopiDtype_t::diopi_dtype_int64);
constexpr uint64_t BoolFloat32      = _MAKE_KEY(diopiDtype_t::diopi_dtype_bool, diopiDtype_t::diopi_dtype_float32);
constexpr uint64_t BoolFloat16      = _MAKE_KEY(diopiDtype_t::diopi_dtype_bool, diopiDtype_t::diopi_dtype_float16);
constexpr uint64_t BoolInt32        = _MAKE_KEY(diopiDtype_t::diopi_dtype_bool, diopiDtype_t::diopi_dtype_int32);
constexpr uint64_t Int64Int32       = _MAKE_KEY(diopiDtype_t::diopi_dtype_int64, diopiDtype_t::diopi_dtype_int32);
constexpr uint64_t Int64Uint32      = _MAKE_KEY(diopiDtype_t::diopi_dtype_int64, diopiDtype_t::diopi_dtype_uint32);
constexpr uint64_t Int64Float16     = _MAKE_KEY(diopiDtype_t::diopi_dtype_int64, diopiDtype_t::diopi_dtype_float16);
constexpr uint64_t Int64Float32     = _MAKE_KEY(diopiDtype_t::diopi_dtype_int64, diopiDtype_t::diopi_dtype_float32);

// special convert (cnnl doesn't support)
constexpr uint64_t BoolInt64        = _MAKE_KEY(diopiDtype_t::diopi_dtype_bool, diopiDtype_t::diopi_dtype_int64);
constexpr uint64_t Int16Int64       = _MAKE_KEY(diopiDtype_t::diopi_dtype_int16, diopiDtype_t::diopi_dtype_int64);
constexpr uint64_t Uint8Bool        = _MAKE_KEY(diopiDtype_t::diopi_dtype_uint8, diopiDtype_t::diopi_dtype_bool);
constexpr uint64_t Int16Bool        = _MAKE_KEY(diopiDtype_t::diopi_dtype_int16, diopiDtype_t::diopi_dtype_bool);
constexpr uint64_t Int64Bool        = _MAKE_KEY(diopiDtype_t::diopi_dtype_int64, diopiDtype_t::diopi_dtype_bool);
constexpr uint64_t Int8Bool         = _MAKE_KEY(diopiDtype_t::diopi_dtype_int8, diopiDtype_t::diopi_dtype_bool);

#define CAST_TYPE(TYPE) \
    { \
        cast_type = TYPE; \
        break; \
    }

#define CAST_TYPE_THROUGH_INT32(TYPE) \
    { \
        input_tr = input_tr.to(ctx, diopiDtype_t::diopi_dtype_int32); \
        cast_type = TYPE; \
        break; \
    }

diopiError_t _diopiCast(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t output, diopiDtype_t from, diopiDtype_t to) {
    // TODO(waiting for dispatch): support broadcast, dealing with uncontiguous
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tr = impl::camb::makeTensor(input);
    auto output_tr = impl::camb::makeTensor(output);

    auto key = _MAKE_KEY(from, to);
    cnnlCastDataType_t cast_type;
    switch (key) {
        // cast once
        case Float64Float32: CAST_TYPE(CNNL_CAST_DOUBLE_TO_FLOAT)
        case Float32Float64: CAST_TYPE(CNNL_CAST_FLOAT_TO_DOUBLE)
        case Float32Float16: CAST_TYPE(CNNL_CAST_FLOAT_TO_HALF)
        case Float32Int64:   CAST_TYPE(CNNL_CAST_FLOAT_TO_INT64)
        case Float32Int32:   CAST_TYPE(CNNL_CAST_FLOAT_TO_INT32)
        case Float32Int16:   CAST_TYPE(CNNL_CAST_FLOAT_TO_INT16)
        case Float32Int8:    CAST_TYPE(CNNL_CAST_FLOAT_TO_INT8)
        case Float32Uint8:   CAST_TYPE(CNNL_CAST_FLOAT_TO_UINT8)
        case Float32Bool:    CAST_TYPE(CNNL_CAST_FLOAT_TO_BOOL)
        case Float16Float32: CAST_TYPE(CNNL_CAST_HALF_TO_FLOAT_INF)
        case Float16Int64:   CAST_TYPE(CNNL_CAST_HALF_TO_INT64)
        case Float16Int32:   CAST_TYPE(CNNL_CAST_HALF_TO_INT32)
        case Float16Int16:   CAST_TYPE(CNNL_CAST_HALF_TO_INT16)
        case Float16Int8:    CAST_TYPE(CNNL_CAST_HALF_TO_INT8)
        case Float16Uint8:   CAST_TYPE(CNNL_CAST_HALF_TO_UINT8)
        case Float16Bool:    CAST_TYPE(CNNL_CAST_HALF_TO_BOOL)
        case Int32Float32:   CAST_TYPE(CNNL_CAST_INT32_TO_FLOAT)
        case Int32Float16:   CAST_TYPE(CNNL_CAST_INT32_TO_HALF)
        case Int32Bool:      CAST_TYPE(CNNL_CAST_INT32_TO_BOOL)
        case Int32Int8:      CAST_TYPE(CNNL_CAST_INT32_TO_INT8)
        case Int32Int16:     CAST_TYPE(CNNL_CAST_INT32_TO_INT16)
        case Int32Int64:     CAST_TYPE(CNNL_CAST_INT32_TO_INT64)
        case Uint32Int64:    CAST_TYPE(CNNL_CAST_UINT32_TO_INT64)
        case Int16Float32:   CAST_TYPE(CNNL_CAST_INT16_TO_FLOAT)
        case Int16Float16:   CAST_TYPE(CNNL_CAST_INT16_TO_HALF)
        case Int16Int32:     CAST_TYPE(CNNL_CAST_INT16_TO_INT32)
        case Int8Float32:    CAST_TYPE(CNNL_CAST_INT8_TO_FLOAT)
        case Int8Float16:    CAST_TYPE(CNNL_CAST_INT8_TO_HALF)
        case Int8Int32:      CAST_TYPE(CNNL_CAST_INT8_TO_INT32)
        case Uint8Float32:   CAST_TYPE(CNNL_CAST_UINT8_TO_FLOAT)
        case Uint8Float16:   CAST_TYPE(CNNL_CAST_UINT8_TO_HALF)
        case Uint8Int32:     CAST_TYPE(CNNL_CAST_UINT8_TO_INT32)
        case Uint8Int64:     CAST_TYPE(CNNL_CAST_UINT8_TO_INT64)
        case BoolFloat32:    CAST_TYPE(CNNL_CAST_BOOL_TO_FLOAT)
        case BoolFloat16:    CAST_TYPE(CNNL_CAST_BOOL_TO_HALF)
        case BoolInt32:      CAST_TYPE(CNNL_CAST_BOOL_TO_INT32)
        case Int64Int32:     CAST_TYPE(CNNL_CAST_INT64_TO_INT32)
        case Int64Uint32:    CAST_TYPE(CNNL_CAST_INT64_TO_UINT32)
        case Int64Float16:   CAST_TYPE(CNNL_CAST_INT64_TO_HALF)
        case Int64Float32:   CAST_TYPE(CNNL_CAST_INT64_TO_FLOAT)

        // cast through middle
        case BoolInt64:      CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_INT64)
        case Int16Int64:     CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_INT64)
        case Uint8Bool:      CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)
        case Int16Bool:      CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)
        case Int64Bool:      CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)
        case Int8Bool:       CAST_TYPE_THROUGH_INT32(CNNL_CAST_INT32_TO_BOOL)

        default:
            set_last_error_string("can not cast from %d to %d at %s:%d ", from, to, __FILE__, __LINE__);
            return diopiErrorOccurred;
    }

    CnnlTensorDesc input_desc(input_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tr, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECKCNNL(cnnlCastDataType(
            handle, input_desc.get(), input_tr.data(), cast_type, output_desc.get(), const_cast<void*>(output_tr.data())));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

#include "common.hpp"

namespace impl {
namespace camb {

using DiopiTensorT = DiopiTensor<diopiTensorHandle_t>;
DiopiTensorT makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize(sizeTmp, 1);
    DiopiTensorT out;
    if (scalar->stype == diopi_dtype_int64) {
        int32_t val = static_cast<int32_t>(scalar->ival);
        DiopiTensorT out(requiresTensor(ctx, sSize, diopi_dtype_int32));
        CnnlTensorDesc descOut(out, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &val, descOut.get(), out.data()));
        return out;
    } else if (scalar->stype == diopi_dtype_float64) {
        float val = static_cast<float>(scalar->fval);
        DiopiTensorT out(requiresTensor(ctx, sSize, diopi_dtype_float32));
        CnnlTensorDesc descOut(out, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &val, descOut.get(), out.data()));
        return out;
    } else {
        assert((void("salar dtype is not float64 or int64"), false));
    }
    return out;
}

}  // namespace camb
}  // namespace impl
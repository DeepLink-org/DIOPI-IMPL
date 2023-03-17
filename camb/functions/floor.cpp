#include <cnrt.h>
#include <diopi/functions.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    auto trInput = makeTensor(input_);
    auto trOut = makeTensor(out);
    std::vector<DiopiTensorT*> pTensors{&trInput};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};

    autoCastTensorType(ctx, pTensors, supportedDtypes);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descOut(trOut, layout);
    DiopiTensorT trOutTmp;
    CnnlTensorDesc descOutTmp;
    if (trInput.dtype() == trOut.dtype()) {
        trOutTmp = trOut;
        descOutTmp = descOut;
    } else {
        trOutTmp = requiresTensor(ctx, vec2diopiSize_t(trOut.shape()), trInput.dtype());
        descOutTmp.set(trOutTmp, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALLCNNL(cnnlFloor(handle, descInput.get(), trInput.data(), descOutTmp.get(), trOutTmp.data()));
    if (trOutTmp.dtype() != trOut.dtype()) {
        dataTypeCast(ctx, trOut, trOutTmp);
    }
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiFloor(ctx, input, input);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

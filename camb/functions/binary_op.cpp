/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

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
extern "C" {

DIOPI_API diopiError_t
diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    auto trInput = DiopiTensor(input);
    auto trOther = DiopiTensor(other);
    auto trOut = DiopiTensor(out);
    std::vector<DiopiTensor*> pTensors{&trInput, &trOther};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32};

    autoCastTensorType(ctx, pTensors, supportedDtypes);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descOther(trOther, layout);
    CnnlTensorDesc descOut;
    DiopiTensor trOutTmp;
    if (trInput.dtype() == trOut.dtype()) {
        trOutTmp = trOut;
        descOut.set(trOut, layout);
    } else {
        trOutTmp = requiresTensor(ctx, vec2diopiSize_t(trOut.shape()), trInput.dtype());
        descOut.set(trOutTmp, CNNL_LAYOUT_ARRAY);
    }

    std::unique_ptr<void, void (*)(void*)> pAlphaIn(malloc(4), free);
    std::unique_ptr<void, void (*)(void*)> pBetaIn(malloc(4), free);
    if (CnnlDataType::isInteger(dtype)) {
        *reinterpret_cast<int32_t*>(pBetaIn.get()) = 0;
        if (alpha->stype <= 7) {
            *reinterpret_cast<int32_t*>(pAlphaIn.get()) = static_cast<int32_t>(alpha->ival);
        } else {
            *reinterpret_cast<int32_t*>(pAlphaIn.get()) = static_cast<int32_t>(static_cast<float>(alpha->fval));
        }
    } else {
        *reinterpret_cast<float*>(pBetaIn.get()) = 0.0f;
        if (alpha->stype <= 7) {
            *reinterpret_cast<float*>(pAlphaIn.get()) = static_cast<float>(static_cast<int32_t>(alpha->ival));
        } else {
            *reinterpret_cast<float*>(pAlphaIn.get()) = static_cast<float>(alpha->fval);
        }
    }
    std::cout << "add:" << *reinterpret_cast<float*>(pAlphaIn.get()) << std::endl;
    DIOPI_CALLCNNL(
        cnnlTransform_v2(handle, CNNL_POINTER_MODE_HOST, pAlphaIn.get(), descOther.get(), trOther.data(), pBetaIn.get(), descOther.get(), trOther.data()));
    const cnnlTensorDescriptor_t inputDescs[2] = {descInput.get(), descOther.get()};
    const void* inputs[2] = {trInput.data(), trOther.data()};
    uint32_t inputNum = 2;
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAddNWorkspaceSize(handle, inputDescs, inputNum, descOut.get(), &workspaceSize));
    auto buff = requiresBuffer(ctx, workspaceSize);
    void* pWorkspace = buff.data();

    DIOPI_CALLCNNL(cnnlAddN_v2(handle, inputDescs, inputs, inputNum, descOut.get(), trOutTmp.data(), pWorkspace, workspaceSize));
    if (trOutTmp.dtype() != trOut.dtype()) {
        dataTypeCast(ctx, trOut, trOutTmp);
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiAdd(ctx, input, input, other, alpha);
    return diopiSuccess;
}

DIOPI_API diopiError_t
diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DiopiTensor trOther = makeTensorFromScalar(ctx, other);
    DIOPI_CALL(diopiAdd(ctx, out, input, static_cast<diopiTensorHandle_t>(trOther), alpha));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t input,
                                                    const diopiScalar_t* other,
                                                    const diopiScalar_t* alpha) {
    diopiAddScalar(ctx, input, input, other, alpha);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    std::cout << "diopiSub" << std::endl;
    std::cout << "int:" << static_cast<int>(alpha->ival) << std::endl;
    std::cout << "float:" << static_cast<float>(alpha->fval) << std::endl;
    diopiScalar_t neg_alpha;
    neg_alpha.stype = alpha->stype;
    if(alpha->stype <= 7) {
        neg_alpha.ival = -alpha->ival;
    }
    else {
        neg_alpha.fval = -alpha->fval;
    }
    diopiAdd(ctx, out, input, other, &neg_alpha);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                   diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    std::cout << "diopiSubInp" << std::endl;
    std::cout << "int:" << static_cast<int>(alpha->ival) << std::endl;
    std::cout << "float:" << static_cast<float>(alpha->fval) << std::endl;
    diopiScalar_t neg_alpha;
    neg_alpha.stype = alpha->stype;
    if(alpha->stype <= 7) {
        neg_alpha.ival = -alpha->ival;
        std::cout << neg_alpha.ival << std::endl;
    }
    else {
        neg_alpha.fval = -alpha->fval;
        std::cout << neg_alpha.fval << std::endl;
    }
    diopiAddInp(ctx, input, other, &neg_alpha); // correct
    // diopiAdd(ctx, input, input, other, alpha);
    // diopiSub(ctx, input, input, other, alpha);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, const diopiScalar_t* alpha) {
    std::cout << "diopiSubScalar" << std::endl;
    std::cout << "int:" << static_cast<int>(alpha->ival) << std::endl;
    std::cout << "float:" << static_cast<float>(alpha->fval) << std::endl;
    diopiScalar_t neg_alpha;
    neg_alpha.stype = alpha->stype;
    if(alpha->stype <= 7) {
        neg_alpha.ival = -alpha->ival;
        // std::cout << neg_alpha.ival << std::endl;
    }
    else {
        neg_alpha.fval = -alpha->fval;
        // std::cout << neg_alpha.fval << std::endl;
    }
    diopiAddScalar(ctx, out, input, other, &neg_alpha);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                         const diopiScalar_t* other, const diopiScalar_t* alpha) {
    std::cout << "diopiSubInpScalar" << std::endl;
    // std::cout << "int:" << static_cast<int>(alpha->ival) << std::endl;
    // std::cout << "float:" << static_cast<float>(alpha->fval) << std::endl;
    //     diopiScalar_t neg_alpha;
    // neg_alpha.stype = alpha->stype;
    // if(alpha->stype <= 7) {
    //     neg_alpha.ival = -alpha->ival;
    // }
    // else {
    //     neg_alpha.fval = -alpha->fval;
    // }
    // diopiAddScalar(ctx, input, input, other, &neg_alpha);
    diopiSubScalar(ctx, input, input, other, alpha);
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl

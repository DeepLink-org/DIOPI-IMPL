#include <cnrt.h>
#include <diopi/functions.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

void print(std::vector<int64_t> vec) {
    std::cout << "vec: ";
    for (auto i : vec) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

void print(diopiSize_t size) {
    std::cout << "diopiSize_t: ";
    for (int i = 0; i < size.len; ++i) {
        std::cout << size.data[i] << ", ";
    }
    std::cout << std::endl;
}

using DiopiTensorT = DiopiTensor<diopiTensorHandle_t>;
DiopiTensorT dataTypeCast(diopiContextHandle_t& ctx, const DiopiTensorT& src, diopiDtype_t destDtype) {
    if (src.dtype() == destDtype) {
        return src;
    }
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiSize_t srcSize = vec2diopiSize_t(src.shape());
    DiopiTensorT dest = requiresTensor(ctx, srcSize, destDtype);
    diopiDtype_t srcDtype = src.dtype();
    cnnlCastDataType_t cnnlCastDtype = gCnnlCastDataTypeMapping[{srcDtype, destDtype}];
    std::cout << "cnnlCastDtype:" << cnnlCastDtype << std::endl;
    DIOPI_CHECK_ABORT(cnnlCastDtype != 0, "data type cast from %d to %d in cnnl is not allown", srcDtype, destDtype);
    CnnlTensorDesc descSrc(src, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc descDest(dest, CNNL_LAYOUT_ARRAY);
    print(srcSize);
    print(src.shape());
    std::cout << "handle: " << handle << std::endl;
    // auto queue = getStream(ctx);
    // cnrtQueueSync(queue);
    // cnrtQueueSync(getStream(ctx));
    DIOPI_CHECKCNNL(cnnlCastDataType(handle, descSrc.get(), (void*)((const_cast<DiopiTensorT&>(src)).data()), cnnlCastDtype, descDest.get(), dest.data()));
    // std::cout << "cnnlCastDtype:" << cnnlCastDtype << "(ok)" << std::endl;
    return dest;
}

void dataTypeCast(diopiContextHandle_t ctx, DiopiTensorT& dest, const DiopiTensorT& src) {
    if (dest.dtype() == src.dtype()) {
        return;
    }
    // check size of dest and src
    std::cout << "++++++dataTypeCast_out++begin++++" << std::endl;
    assert((void("the shapes of src and dest are not equal"), src.shape().size() != 0 && src.shape() == dest.shape()));
    print(src.shape());
    print(dest.shape());
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiDtype_t srcDtype = src.dtype();
    diopiDtype_t destDtype = dest.dtype();
    CnnlTensorDesc descSrc(src, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc descDest(dest, CNNL_LAYOUT_ARRAY);
    cnnlCastDataType_t cnnlCastDtype = gCnnlCastDataTypeMapping[{srcDtype, destDtype}];
    std::cout << "cnnlCastDtype:" << cnnlCastDtype << std::endl;
    cnnlHandle_t handleTmp;
    cnnlCreate(&handleTmp);

    cnrtQueue_t q0;
    cnrtQueueCreate(&q0);
    cnnlSetQueue(handleTmp, q0);

    cnrtQueueSync(getStream(ctx));
    //std::cout << camb_get_last_error_string() << std::endl;
    DIOPI_CHECKCNNL(cnnlCastDataType(handleTmp, descSrc.get(), (void*)(const_cast<DiopiTensorT&>(src).data()), (cnnlCastDataType_t)cnnlCastDtype, descDest.get(), dest.data()));
    cnrtQueueSync(q0);

    std::cout << "cnnlCastDtype:" << cnnlCastDtype << "(ok)" << std::endl;
    std::cout << "++++++dataTypeCast_out++end++++" << std::endl;
    return;
}

DiopiTensorT makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize(sizeTmp, 1);
    DiopiTensorT out;
    if (scalar->stype == diopi_dtype_int64) {
        int32_t val = (int32_t)scalar->ival;
        DiopiTensorT out(requiresTensor(ctx, sSize, diopi_dtype_int32));
        CnnlTensorDesc descOut(out, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &val, descOut.get(), out.data()));
        return out;
    } else if (scalar->stype == diopi_dtype_float64) {
        float val = (float)scalar->fval;
        DiopiTensorT out(requiresTensor(ctx, sSize, diopi_dtype_float32));
        CnnlTensorDesc descOut(out, CNNL_LAYOUT_ARRAY);
        DIOPI_CHECKCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &val, descOut.get(), out.data()));
        return out;
    } else {
        assert((void("salar dtype is not float64 or int64"), false));
    }
    return out;
}

diopiDtype_t choiceDtype(const std::set<diopiDtype_t>& opSupportedDtypes) {
    if (opSupportedDtypes.find(diopi_dtype_float32) != opSupportedDtypes.end()) {
        return diopi_dtype_float32;
    }
    if (opSupportedDtypes.find(diopi_dtype_float16) != opSupportedDtypes.end()) {
        return diopi_dtype_float16;
    }
    if (opSupportedDtypes.find(diopi_dtype_int32) != opSupportedDtypes.end()) {
        return diopi_dtype_int32;
    }
    if (opSupportedDtypes.find(diopi_dtype_int16) != opSupportedDtypes.end()) {
        return diopi_dtype_int16;
    }
    if (opSupportedDtypes.find(diopi_dtype_int8) != opSupportedDtypes.end()) {
        return diopi_dtype_int8;
    }
    if (opSupportedDtypes.find(diopi_dtype_bool) != opSupportedDtypes.end()) {
        return diopi_dtype_bool;
    }
    assert((void("this operator does not support bool, int8, int16, int32, float16, float32"), false));
    return diopi_dtype_int64;  // just for return a value
}

void autoCastTensorType(diopiContextHandle_t ctx, std::vector<DiopiTensorT*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype) {
    // std::multimap<diopiDtype_t, DiopiTensorT*> dtypeAndTensorPtrs;
    std::set<diopiDtype_t> dtypeAndTensorPtrs;
    diopiDtype_t targetType = diopi_dtype_float32;
    for (const auto& pTensor : pTensors) {
        dtypeAndTensorPtrs.insert(pTensor->dtype());
    }
    if (dtypeAndTensorPtrs.find(diopi_dtype_bool) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_bool) == opSupportedDtype.end()) {  // not support bool
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into bool
            targetType = diopi_dtype_bool;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_float64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_float32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float32) == opSupportedDtype.end()) {  // not support float32
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into float32
            targetType = diopi_dtype_float32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_float16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float16) == opSupportedDtype.end()) {  // not support float16
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into float16
            targetType = diopi_dtype_float16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_int32) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int32) == opSupportedDtype.end()) {  // not support int32
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into int32
            targetType = diopi_dtype_int32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int16) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int16) == opSupportedDtype.end()) {  // not support int16
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into int16
            targetType = diopi_dtype_int16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int8) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint8) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int8) == opSupportedDtype.end()) {  // not support int8
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into int8
            targetType = diopi_dtype_int8;
        }
    } else {
        assert((void("tensor's dtype error, can't be cast"), false));
    }
    for (auto pTensor : pTensors) {
        std::cout << "targetType: " << targetType << std::endl;
        *pTensor = dataTypeCast(ctx, *pTensor, targetType);
    }
}

extern "C" DIOPI_API diopiError_t
diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    static int cnt = 0;
    cnt+=1;
    std::cout << "=========begin=============" << cnt <<std::endl;
    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    diopiTensorHandle_t other_ = diopiTensorHandle_t(other);
    auto trInput = makeTensor(input_);
    auto trOther = makeTensor(other_);
    auto trOut = makeTensor(out);
    std::vector<DiopiTensorT*> pTensors{&trInput, &trOther};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32};

    std::cout << "inputs[0].dtype():" << trInput.dtype() << "  inputs[1].dtype():" << trOther.dtype() << " ouput.dtype():" << trOut.dtype() << std::endl;
    autoCastTensorType(ctx, pTensors, supportedDtypes);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descOther(trOther, layout);
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
    DIOPI_CALLCNNL(
        cnnlTransform_v2(handle, CNNL_POINTER_MODE_HOST, pAlphaIn.get(), descOther.get(), trOther.data(), pBetaIn.get(), descOther.get(), trOther.data()));
    const cnnlTensorDescriptor_t inputDescs[2] = {descInput.get(), descOther.get()};
    const void* inputs[2] = {trInput.data(), trOther.data()};
    uint32_t inputNum = 2;
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAddNWorkspaceSize(handle, inputDescs, inputNum, descOutTmp.get(), &workspaceSize));
    auto buff = requiresBuffer(ctx, workspaceSize);
    void* pWorkspace = buff.data_ptr();

    DIOPI_CALLCNNL(cnnlAddN_v2(handle, inputDescs, inputs, inputNum, descOutTmp.get(), trOutTmp.data(), pWorkspace, workspaceSize));
    if (trOutTmp.dtype() != trOut.dtype()) {
        dataTypeCast(ctx, trOut, trOutTmp);
    }
    std::cout << "+++++++++++++end==========: " << std::endl;
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiAdd(ctx, input, input, other, alpha);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t
diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    std::cout << "add sclar ===" << std::endl;
    DiopiTensorT trOther = makeTensorFromScalar(ctx, other);
    DIOPI_CALL(diopiAdd(ctx, out, input, trOther, alpha));
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t input,
                                                    const diopiScalar_t* other,
                                                    const diopiScalar_t* alpha) {
    std::cout << "add sclar ===" << std::endl;
    diopiAddScalar(ctx, input, input, other, alpha);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
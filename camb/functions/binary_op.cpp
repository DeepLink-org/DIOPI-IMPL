#include <cnrt.h>
#include <diopi/functions.h>

#include <vector>
#include <iostream>
#include <memory>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    diopiTensorHandle_t other_ = diopiTensorHandle_t(other);
    auto trInput = makeTensor(input_);
    auto trOther = makeTensor(other_);
    auto trOut = makeTensor(out);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput,layout);
    CnnlTensorDesc descOther(trOther,layout);
    CnnlTensorDesc descOut(trOut,layout);

    std::vector<int32_t> inputShape = trInput.shape();
    int inputShapeSize = inputShape.size();
    // if (inputShapeSize == 0) {
    //     inputShapeSize = 1;
    //     inputShape.push_back(1);
    // }

    std::vector<int32_t> otherShape = trOther.shape();
    int otherShapeSize = otherShape.size();
    // if (otherShapeSize == 0) {
    //     otherShapeSize = 1;
    //     otherShape.push_back(1);
    // }

    std::vector<int32_t> outShape = trOut.shape();
    int outShapeSize = outShape.size();
    // if (outShapeSize == 0) {
    //     outShapeSize = 1;
    //     outShape.push_back(1);
    // }

    std::unique_ptr<void, void(*)(void*)> pAlphaIn (malloc(4), free);
    std::unique_ptr<void, void(*)(void*)> pBetaIn (malloc(4), free);
    if (dtype >= 3 && dtype <= 13)  {
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
    DIOPI_CALLCNNL(cnnlTransform(handle, pAlphaIn.get(), descOther.get(), trOther.data(), pBetaIn.get(), trOther.data()));
    const cnnlTensorDescriptor_t inputDescs[2] = {descInput.get(), descOther.get()};
    const void* inputs[2] = {trInput.data(), trOther.data()};
    uint32_t inputNum = 2;
    std::cout << "add" << "\t inputShapeSize: " << inputShapeSize << "\t otherShapeSize: " << otherShapeSize << "\t outShapeSize: " <<  outShapeSize << std::endl;
    //DIOPI_CALLCNNL(cnnlAddN(handle, inputDescs, inputs, inputNum, descOut.get(), trOut.data()));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAddNWorkspaceSize(handle, inputDescs, inputNum, descOut.get(), &workspaceSize));
    std::cout << "workspaceSize:" << workspaceSize << std::endl;
    diopiTensorHandle_t pBuff;
    diopiRequireBuffer(ctx, &pBuff, workspaceSize, diopi_device);
    void *pWorkspace = nullptr;
    diopiGetTensorData(&pBuff, &pWorkspace);
    DIOPI_CALLCNNL(cnnlAddN_v2(handle,inputDescs, inputs, inputNum, descOut.get(), trOut.data(), pWorkspace, workspaceSize));

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiAdd(ctx, input, input,other,alpha);
}

// DIOPI_API diopiError_t
// diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
//     auto stream = getStream(ctx);
//     diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
//     auto trInput = makeTensor(input_);
//     auto trOutput = makeTensor(out);

//     CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
//     cnnlHandle_t handle = CnnlHandle.get();
//     DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

//     cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
//     cnnlDataType_t dtype;
//     DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

//     CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescInput;
//     cnnlTensorDescriptor_t descInput = CnnlDescInput.get();

//     CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescOut;
//     cnnlTensorDescriptor_t descOut = CnnlDescOut.get();

//     int dimNbInput = trInput.shape().len;
//     std::vector<int> dimSizeInput(dimNbInput);
//     if (dimNbInput == 0) {
//         dimNbInput = 1;
//         dimSizeInput.push_back(1);
//     } else {
//         for (int i = 0; i < dimNbInput; ++i) {
//             dimSizeInput[i] = trInput.shape().data[i];
//         }
//     }

//     int dimNbOut = trOutput.shape().len;
//     std::vector<int> dimSizeOut(dimNbOut);
//     if (dimNbOut == 0) {
//         dimNbOut = 1;
//         dimSizeOut.push_back(1);
//     } else {
//         for (int i = 0; i < dimNbOut; ++i) {
//             dimSizeOut[i] = trOutput.shape().data[i];
//         }
//     }

//     float value;
//     if (other->stype <= 7 && alpha->stype <= 7) {
//         value = static_cast<float>(static_cast<int32_t>(other->ival) * static_cast<int32_t>(alpha->ival));
//         std::cout << "other: " << other->ival << "\t alpha: " << alpha->ival << std::endl;
//     } else if (other->stype <= 7 && alpha->stype > 7) {
//         value = static_cast<int32_t>(other->ival) * static_cast<float>(alpha->fval);
//         std::cout << "other: " << other->ival << "\t alpha: " << alpha->fval << std::endl;
//     } else if (other->stype > 7 && alpha->stype <= 7) {
//         value = static_cast<float>(other->fval) * static_cast<int32_t>(alpha->ival);
//         std::cout << "other: " << other->fval << "\t alpha: " << alpha->ival << std::endl;
//     } else {
//         value = static_cast<float>(other->fval) * static_cast<float>(alpha->fval);
//         std::cout << "other: " << other->fval << "\t alpha: " << alpha->fval << std::endl;
//     }

//     DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descInput, layout, dtype, dimNbInput, dimSizeInput.data()));
//     DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descOut, layout, dtype, dimNbOut, dimSizeOut.data()));
//     DIOPI_CALLCNNL(cnnlFill(handle, value, descOut, trOutput.data()));

//     const cnnlTensorDescriptor_t input_descs[2] = {descInput, descOut};
//     const void* inputs[2] = {trInput.data(), trOutput.data()};
//     uint32_t input_num = 2;
//     std::cout << "add_scalar"
//               << "\t dimSizeInput: " << dimNbInput << "\t dimSizeOut: " << dimNbOut << std::endl;
//     std::cout << "dtype: " << dtype << "\t value: " << value << std::endl;
//     DIOPI_CALLCNNL(cnnlAddN(handle, input_descs, inputs, input_num, descOut, trOutput.data()));
//     return diopiSuccess;
// }

// DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
//     auto stream = getStream(ctx);
//     diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
//     diopiTensorHandle_t other_ = diopiTensorHandle_t(input);
//     auto trInput = makeTensor(input_);
//     auto trOther = makeTensor(other_);

//     CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
//     cnnlHandle_t handle = CnnlHandle.get();
//     DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

//     cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
//     cnnlDataType_t dtype;
//     DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

//     CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescInput;
//     cnnlTensorDescriptor_t descInput = CnnlDescInput.get();

//     CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDescOther;
//     cnnlTensorDescriptor_t descOther = CnnlDescOther.get();

//     int dimNbInput = trInput.shape().len;
//     std::vector<int> dimSizeInput(dimNbInput);
//     if (dimNbInput == 0) {
//         dimNbInput = 1;
//         dimSizeInput.push_back(1);
//     } else {
//         for (int i = 0; i < dimNbInput; ++i) {
//             dimSizeInput[i] = trInput.shape().data[i];
//         }
//     }

//     float value;
//     if (other->stype <= 7 && alpha->stype <= 7) {
//         value = static_cast<float>(static_cast<int32_t>(other->ival) * static_cast<int32_t>(alpha->ival));
//         std::cout << "other: " << other->ival << "\t alpha: " << alpha->ival << std::endl;
//     } else if (other->stype <= 7 && alpha->stype > 7) {
//         value = static_cast<int32_t>(other->ival) * static_cast<float>(alpha->fval);
//         std::cout << "other: " << other->ival << "\t alpha: " << alpha->fval << std::endl;
//     } else if (other->stype > 7 && alpha->stype <= 7) {
//         value = static_cast<float>(other->fval) * static_cast<int32_t>(alpha->ival);
//         std::cout << "other: " << other->fval << "\t alpha: " << alpha->ival << std::endl;
//     } else {
//         value = static_cast<float>(other->fval) * static_cast<float>(alpha->fval);
//         std::cout << "other: " << other->fval << "\t alpha: " << alpha->fval << std::endl;
//     }

//     DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descInput, layout, dtype, dimNbInput, dimSizeInput.data()));
//     DIOPI_CALLCNNL(cnnlSetTensorDescriptor(descOther, layout, dtype, dimNbInput, dimSizeInput.data()));
//     DIOPI_CALLCNNL(cnnlFill(handle, value, descOther, trOther.data()));

//     const cnnlTensorDescriptor_t input_descs[2] = {descInput, descOther};
//     const void* inputs[2] = {trInput.data(), trOther.data()};
//     uint32_t input_num = 2;
//     std::cout << "add_scalar_inp"
//               << "\t dimSizeInput: " << dimNbInput << "\t dimSizeOut: " << dimNbInput << std::endl;
//     std::cout << "dtype: " << dtype << "\t value: " << value << std::endl;
//     DIOPI_CALLCNNL(cnnlAddN(handle, input_descs, inputs, input_num, descInput, trInput.data()));
//     return diopiSuccess;
// }

}  // extern "C"

}  // namespace camb
}  // namespace impl
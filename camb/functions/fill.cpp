#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto trInput = impl::camb::makeTensor(input);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDesc;
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));
    cnnlTensorDescriptor_t desc = CnnlDesc.get();

    std::vector<int32_t> strides = trInput.stride();
    std::vector<int32_t> shape = trInput.shape();
    if (strides.size() == 0) {
        shape.push_back(1);
        strides.push_back(1);
    }

    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }

    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(desc, layout, dtype, shape.size(), shape.data(), strides.data()));
    DIOPI_CALLCNNL(cnnlFill(handle, val, desc, trInput.data()));
    return diopiSuccess;
}

}  // extern "C"

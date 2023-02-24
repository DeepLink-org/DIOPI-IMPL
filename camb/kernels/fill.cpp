#include "../helper.hpp"

extern "C" diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto stream  = impl::camb::getStream(ctx);
    auto trInput = impl::camb::makeTensor(input);

    HandleGuard handle_guard;
    cnnlHandle_t handle = handle_guard.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));
    if(CNNL_DTYPE_DOUBLE == dtype){
        return diopiDtypeNotSupported;
    }
    TensorDescGuard desc_guard;
    cnnlTensorDescriptor_t desc = desc_guard.get();
    std::vector<int> shape = trInput.shape();
    int dimNb = trInput.shape_len();
    std::vector<int> dimStrides(dimNb, 1);
    std::vector<int> dimSize(dimNb);
    std::vector<int> stride = trInput.stride();

    if (dimNb == 0) {
        dimNb = 1;
        dimSize.push_back(1);
        dimStrides.push_back(1);
    } else {
        for (int i = 0; i < dimNb; ++i) {
            dimSize[i] = shape[i];
        }
        if (dimNb > 0) {
            for (int i = 0; i < dimNb; ++i) {
                dimStrides[i] = stride[i];
            }
        }
    }

    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }
    cnnlPointerMode_t point_mode = CNNL_POINTER_MODE_HOST;
    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(desc, layout, dtype, dimNb,dimSize.data(), dimStrides.data()));
    DIOPI_CALLCNNL(cnnlFill_v3(handle, point_mode, &val, desc, trInput.data()));
    return diopiSuccess;
}


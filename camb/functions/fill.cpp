<<<<<<< HEAD
#include <vector>

#include <diopi/functions.h>

=======
#include <diopi/functions.h>

#include <vector>

>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb
#include "../cnnl_helper.hpp"

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
<<<<<<< HEAD
    auto stream  = impl::camb::getStream(ctx);
=======
    auto stream = impl::camb::getStream(ctx);
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb
    auto trInput = impl::camb::makeTensor(input);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

<<<<<<< HEAD
    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDesc;
=======
    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDesc;
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));
    cnnlTensorDescriptor_t desc = CnnlDesc.get();

    diopiSize_t shape = trInput.shape();
    int dimNb = shape.len;
    std::vector<int> dimStrides(dimNb, 1);
    std::vector<int> dimSize(dimNb);
    diopiSize_t stride = trInput.stride();

    if (dimNb == 0) {
        dimNb = 1;
        dimSize.push_back(1);
        dimStrides.push_back(1);
    } else {
        for (int i = 0; i < dimNb; ++i) {
            dimSize[i] = shape.data[i];
        }
        if (dimNb > 0) {
            for (int i = 0; i < dimNb; ++i) {
                dimStrides[i] = stride.data[i];
            }
        }
    }

    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }

<<<<<<< HEAD
    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(desc, layout, dtype, dimNb,
        dimSize.data(), dimStrides.data()));
=======
    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(desc, layout, dtype, dimNb, dimSize.data(), dimStrides.data()));
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb
    DIOPI_CALLCNNL(cnnlFill(handle, val, desc, trInput.data()));
    return diopiSuccess;
}

<<<<<<< HEAD
}  // extern "C"
=======
}  // extern "C"
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb

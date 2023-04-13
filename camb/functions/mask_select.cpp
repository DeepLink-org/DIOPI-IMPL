#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t nonzeroCount(diopiContextHandle_t ctx, DiopiTensor input_tensor, DiopiTensor* num_true);

DIOPI_API diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor mask_tensor(mask);

    std::vector<DiopiTensor*> pmask{&mask_tensor};
    std::set<diopiDtype_t> mask_dtypes{diopi_dtype_bool, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pmask, mask_dtypes));
    // When the data type of masked tensor is not bool, the data type of input tensor must be same with the data type of the masked tensor.
    if (mask_tensor.dtype() != diopi_dtype_bool) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, mask_tensor.dtype()))
    } else {
        std::vector<DiopiTensor*> pinput{&input_tensor};
        std::set<diopiDtype_t> input_dtypes{
            diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
        DIOPI_CALL(autoCastTensorType(ctx, pinput, input_dtypes));
    }

    // count the non_zero elements in mask_tensor
    DiopiTensor num_true;
    DIOPI_CALL(nonzeroCount(ctx, mask_tensor, &num_true));
    syncStreamInCtx(ctx);
    uint32_t count = 0;
    cnrtMemcpy(&count, num_true.data(), sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST);
    // requires the out_tensor according to num_true
    std::cout << "**************************" << std::endl;
    std::cout << "count = " << count << std::endl;
    std::vector<int64_t> shape(1, int64_t(count));
    auto out_tensor = requiresTensor(ctx, shape, input_tensor.dtype());
    std::cout << "input_tensor.numel() = " << input_tensor.numel() << std::endl;
    std::cout << "out_tensor.numel() = " << out_tensor.numel() << std::endl;
    std::cout << "input_tensor.dtype() = " << input_tensor.dtype() << std::endl;
    std::cout << "out_tensor.dtype() = " << out_tensor.dtype() << std::endl;

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mask_desc(mask_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    cnnlMaskedOp_t masked_mode = CNNL_MASKED_SELECT;

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle, masked_mode, input_desc.get(), mask_desc.get(), nullptr, out_desc.get(), &workspace_size));
    std::cout << "workspace_size = " << workspace_size << std::endl;
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    // cast the num_true from int32 to uint32
    DIOPI_CALL(dataTypeCast(ctx, num_true, diopi_dtype_int64));
    DIOPI_CALL(dataTypeCast(ctx, num_true, diopi_dtype_uint32));
    void* temp = num_true.data();
    DIOPI_CALLCNNL(cnnlMasked_v4(handle,
                                 masked_mode,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 mask_desc.get(),
                                 mask_tensor.data(),
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 workspace,
                                 workspace_size,
                                 out_desc.get(), // [cnnlMasked_v3] Failed to launch kernel  output size is too small 
                                 out_tensor.data(),
                                 static_cast<uint32_t*>(temp)));
    *out = diopiTensorHandle_t(out_tensor);
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
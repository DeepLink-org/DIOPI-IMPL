#include "../common/common.hpp"
#include "../common/cnnl_scalar.hpp"
#include "../common/float16.hpp"
#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                      diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor = DiopiTensor(input);
    DiopiTensor grad_tensor = DiopiTensor(grad);
    DiopiTensor square_avg_tensor = DiopiTensor(square_avg);
    DiopiTensor acc_delta_tensor = DiopiTensor(acc_delta);

    DiopiTensor input_casted = input_tensor;
    DiopiTensor grad_casted = grad_tensor;
    DiopiTensor square_avg_casted = square_avg_tensor;
    DiopiTensor acc_delta_casted = acc_delta_tensor;

    std::vector<DiopiTensor*> tensors{&input_casted, &grad_casted, &square_avg_casted, &acc_delta_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    // a = a * scale_a + b * scale_b;
    auto add_mul_func = [&](auto &a, float scale_a, auto b, float scale_b) {
        size_t workspace_size;
        std::vector<int> shape;
        shape.push_back(a.numel());
        CnnlTensorDesc a_desc, b_desc;
        DIOPI_CALL(a_desc.set(a, CNNL_LAYOUT_ARRAY, shape));
        DIOPI_CALL(b_desc.set(b, CNNL_LAYOUT_ARRAY, shape));

        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, b_desc.get(), a_desc.get(), &workspace_size));

        void *workspace = nullptr;
        if (workspace_size != 0) {
            workspace = requiresBuffer(ctx, workspace_size).data();
        }

        DIOPI_CALLCNNL(cnnlBiasAdd(handle, &scale_b, b_desc.get(), b.data(), workspace, workspace_size, &scale_a, a_desc.get(), a.data()));
        return diopiSuccess;
    };


    if(weight_decay != 0){
        DIOPI_CALL(add_mul_func(grad_casted, 1.0, input_casted, weight_decay));
    }

    CnnlTensorDesc input_desc(input_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc square_avg_desc(square_avg_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc acc_delta_desc(acc_delta_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_desc(grad_casted, CNNL_LAYOUT_ARRAY);

    CnnlScalar mlu_lr, mlu_rho, mlu_eps;
    if (input_casted.dtype() == diopi_dtype_float32) {
        mlu_lr.set(lr);
        mlu_rho.set(rho);
        mlu_eps.set(eps);
    } else {
        half_float::half lr_half(lr);
        half_float::half rho_half(rho);
        half_float::half eps_half(eps);
        mlu_lr.set(lr_half);
        mlu_rho.set(rho_half);
        mlu_eps.set(eps_half);
    }
    DIOPI_CALLCNNL(cnnlApplyAdadelta(handle,
                                     input_desc.get(),
                                     input_casted.data(),
                                     square_avg_desc.get(),
                                     square_avg_casted.data(),
                                     acc_delta_desc.get(),
                                     acc_delta_casted.data(),
                                     grad_desc.get(),
                                     grad_casted.data(),
                                     mlu_lr.data(),
                                     mlu_rho.data(),
                                     mlu_eps.data()));

    DIOPI_CALL(dataTypeCast(ctx, input_tensor, input_casted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

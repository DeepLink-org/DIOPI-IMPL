// #include "../common/cnnl_scalar.hpp"
// #include "../common/common.hpp"
// #include "../common/float16.hpp"
// #include "../diopi_helper.hpp"

// namespace impl {
// namespace camb {

// extern "C" diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
//                                    diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
//                                    float weight_decay, int64_t step, bool amsgrad) {
//     cnnlHandle_t handle = cnnlHandlePool.get(ctx);
// }

// // extern "C" diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
// //                                       diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {
// //     cnnlHandle_t handle = cnnlHandlePool.get(ctx);
// //     DiopiTensor input_tensor = DiopiTensor(input);
// //     DiopiTensor grad_tensor = DiopiTensor(grad);
// //     DiopiTensor square_avg_tensor = DiopiTensor(square_avg);
// //     DiopiTensor acc_delta_tensor = DiopiTensor(acc_delta);

// //     DiopiTensor input_casted = input_tensor;
// //     DiopiTensor grad_casted = grad_tensor;
// //     DiopiTensor square_avg_casted = square_avg_tensor;
// //     DiopiTensor acc_delta_casted = acc_delta_tensor;

// //     std::vector<DiopiTensor*> tensors{&input_casted, &grad_casted, &square_avg_casted, &acc_delta_casted};
// //     DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

// //     CnnlTensorDesc input_desc(input_casted, CNNL_LAYOUT_ARRAY);
// //     CnnlTensorDesc square_avg_desc(square_avg_casted, CNNL_LAYOUT_ARRAY);
// //     CnnlTensorDesc acc_delta_desc(acc_delta_casted, CNNL_LAYOUT_ARRAY);
// //     CnnlTensorDesc grad_desc(grad_casted, CNNL_LAYOUT_ARRAY);

// //     CnnlScalar mlu_lr, mlu_rho, mlu_eps;
// //     if (input_casted.dtype() == diopi_dtype_float32) {
// //         mlu_lr.set(lr);
// //         mlu_rho.set(rho);
// //         mlu_eps.set(eps);
// //     } else {
// //         half_float::half lr_half(lr);
// //         half_float::half rho_half(rho);
// //         half_float::half eps_half(eps);
// //         mlu_lr.set(lr_half);
// //         mlu_rho.set(rho_half);
// //         mlu_eps.set(eps_half);
// //     }
// //     DIOPI_CALLCNNL(cnnlApplyAdadelta(handle,
// //                                      input_desc.get(),
// //                                      input_casted.data(),
// //                                      square_avg_desc.get(),
// //                                      square_avg_casted.data(),
// //                                      acc_delta_desc.get(),
// //                                      acc_delta_casted.data(),
// //                                      grad_desc.get(),
// //                                      grad_casted.data(),
// //                                      mlu_lr.data(),
// //                                      mlu_rho.data(),
// //                                      mlu_eps.data()));

// //     DIOPI_CALL(dataTypeCast(ctx, input_tensor, input_casted));
// //     return diopiSuccess;
// // }

// }  // namespace camb
// }  // namespace impl

#include "../common/cnnl_scalar.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"
#include "../diopi_helper.hpp"

namespace impl {
namespace camb {


bool bang_fused_adam(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList grads,
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    int64_t step,
    int64_t mode,
    int64_t bias_correction,
    double weight_decay) {
  auto queue = getCurQueue();
  auto tensor_num = grads.size();

  float beta1_correction_recip = 1;
  float beta2_correction_recip = 1;
  if (bias_correction == 1) {
    beta1_correction_recip = 1 / (1 - std::pow(beta1, step));
    beta2_correction_recip = 1 / (1 - std::pow(beta2, step));
  }

  float epsilon_correction = epsilon / std::sqrt(beta2_correction_recip);
  float learning_rate_correction =
      learning_rate * beta1_correction_recip / std::sqrt(beta2_correction_recip);
  float weight_decay_correction = 1 - learning_rate * weight_decay;
  cnrtDataType_t cnrt_type =
      fromCnnlType2CnrtType(getCnnlType(getMluTensorImpl(grads[0])));

  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  uint32_t union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.x = core_dim;
  k_dim.y = union_number;
  k_dim.z = 1;

  AddressList g, m, v, p;
  SizeList sizes;
  int tensor_count = 0;
  std::vector<std::vector<at::Tensor>> contiguous_tensors_list;
  for (int64_t i = 0; i < tensor_num; ++i) {
    at::Tensor grad = grads[i];
    at::Tensor exp_avg = exp_avgs[i];
    at::Tensor exp_avg_sq = exp_avg_sqs[i];
    at::Tensor param = params[i];
    int64_t num_elements = grad.numel();
    std::vector<at::Tensor> contiguous_tensors;

    auto memory_format = param.suggest_memory_format();
    auto grad_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(grad, memory_format);
    auto exp_avg_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(exp_avg, memory_format);
    auto exp_avg_sq_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(exp_avg_sq, memory_format);
    auto param_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(param, memory_format);

    contiguous_tensors.push_back(exp_avg_contiguous);
    contiguous_tensors.push_back(exp_avg_sq_contiguous);
    contiguous_tensors.push_back(param_contiguous);
    contiguous_tensors_list.push_back(contiguous_tensors);

    auto grad_ptr = getMluTensorImpl(grad_contiguous)->cnnlMalloc();
    auto exp_avg_ptr = getMluTensorImpl(exp_avg_contiguous)->cnnlMalloc();
    auto exp_avg_sq_ptr = getMluTensorImpl(exp_avg_sq_contiguous)->cnnlMalloc();
    auto param_ptr = getMluTensorImpl(param_contiguous)->cnnlMalloc();

    if (num_elements == 0) {
      CNLOG(INFO) << "Adam: Skip zero element tensor.";
      continue;
    }

    g.addresses[tensor_count] = grad_ptr;
    m.addresses[tensor_count] = exp_avg_ptr;
    v.addresses[tensor_count] = exp_avg_sq_ptr;
    p.addresses[tensor_count] = param_ptr;
    sizes.sizes[tensor_count] = num_elements;

    ++tensor_count;
    if (tensor_count == MAX_TENSOR_NUM) {
      bang_fused_adam_internal(
          g, m, v, p, sizes, tensor_count, beta1, beta2,
          epsilon_correction, learning_rate_correction,
          mode, weight_decay, weight_decay_correction, 
          k_dim, k_type, queue, cnrt_type);
      tensor_count = 0;
    }
  }
  if (tensor_count != 0) {
    bang_fused_adam_internal(
        g, m, v, p, sizes, tensor_count, beta1, beta2,
        epsilon_correction, learning_rate_correction,
        mode, weight_decay, weight_decay_correction,
        k_dim, k_type, queue, cnrt_type);
  }

  for (int64_t i = 0; i < tensor_num; ++i) {
    if(torch_mlu::cnnl::ops::is_copy_necessary(exp_avgs[i], contiguous_tensors_list[i][0])) {
      exp_avgs[i].copy_(contiguous_tensors_list[i][0]);
    }
    if(torch_mlu::cnnl::ops::is_copy_necessary(exp_avg_sqs[i], contiguous_tensors_list[i][1])) {
      exp_avg_sqs[i].copy_(contiguous_tensors_list[i][1]);
    }
    if(torch_mlu::cnnl::ops::is_copy_necessary(params[i], contiguous_tensors_list[i][2])) {
      params[i].copy_(contiguous_tensors_list[i][2]);
    }
  }

  return true;
}



extern "C" diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                  diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int64_t step, bool amsgrad) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor = DiopiTensor(input);
    DiopiTensor grad_tensor = DiopiTensor(grad);
    DiopiTensor exp_avg_tensor = DiopiTensor(exp_avg);
    DiopiTensor exp_avg_sq_tensor = DiopiTensor(exp_avg_sq);
    DiopiTensor max_exp_avg_sq_tensor = DiopiTensor(max_exp_avg_sq);

    DiopiTensor input_casted = input_tensor;
    DiopiTensor grad_casted = grad_tensor;
    DiopiTensor exp_avg_casted = exp_avg_tensor;
    DiopiTensor exp_avg_sq_casted = exp_avg_sq_tensor;
    DiopiTensor max_exp_avg_sq_casted = max_exp_avg_sq_tensor;

    std::vector<DiopiTensor *> tensors{&input_casted, &grad_casted, &exp_avg_casted, &exp_avg_sq_casted, &max_exp_avg_sq_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    CnnlTensorDesc input_desc(input_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_desc(grad_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc exp_avg_desc(exp_avg_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc exp_avg_sq_desc(exp_avg_sq_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc max_exp_avg_sq_desc(max_exp_avg_sq_casted, CNNL_LAYOUT_ARRAY);

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

    if (weight_decay != 0) {
        DIOPI_CALL(add_mul_func(grad_casted, 1.0, input_casted, weight_decay));
    }

    CnnlScalar mlu_lr, mlu_beta1, mlu_beta2, mlu_beta1_power, mlu_beta2_power, mlu_epsilon;
    if (input_casted.dtype() == diopi_dtype_float32) {
        mlu_lr.set(lr);
        mlu_beta1.set(beta1);
        mlu_beta2.set(beta2);

        mlu_beta1_power.set(beta1);
        mlu_beta2_power.set(beta2);
        mlu_epsilon.set(eps);

    } else {
        half_float::half lr_half(lr);
        half_float::half beta1_half(beta1);
        half_float::half beta2_half(beta2);
        half_float::half eps_half(eps);

        mlu_lr.set(lr_half);
        mlu_beta1.set(beta1_half);
        mlu_beta2.set(beta2_half);

        mlu_beta1_power.set(beta1_half);
        mlu_beta2_power.set(beta2_half);
        mlu_epsilon.set(eps_half);
    }

    AddressList grad_addr{grad_casted.data()};

    

    bang_fused_adam_internal(
        AddressList{grad_casted.data()},

    )

    // bool use_nesterov = amsgrad;
    // DIOPI_CALLCNNL(cnnlApplyAdam(handle,
    //                              input_desc.get(),
    //                              input_casted.data(),
    //                              exp_avg_desc.get(),
    //                              exp_avg_casted.data(),
    //                              exp_avg_sq_desc.get(),
    //                              exp_avg_sq_casted.data(),
    //                              grad_desc.get(),
    //                              grad_casted.data(),
    //                              mlu_lr.data(),
    //                              mlu_beta1.data(),
    //                              mlu_beta2.data(),
    //                              mlu_beta1_power.data(),
    //                              mlu_beta2_power.data(),
    //                              mlu_epsilon.data(),
    //                              use_nesterov));
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl

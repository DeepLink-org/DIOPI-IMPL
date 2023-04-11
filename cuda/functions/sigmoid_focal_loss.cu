#include <cuda_runtime.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include <stdio.h>
#include<float.h>
#include <iostream>
#include <vector>

#include "../cuda_helper.hpp"
#include "../helper.hpp"

namespace impl {

namespace cuda {


template <typename T>
__global__ void sigmoid_focal_loss_forward_cuda_kernel_diopi(
    const int nthreads, const void* input_, const int64_t* target, const void* weight_,
    void* output_, const T gamma, const T alpha, const int num_classes) {
        const T* input =  static_cast<const T*>(input_);
        T* output = static_cast<T*>(output_);
        const T* weight = static_cast<const T*>(weight_);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + expf(-input[index]));

    // (1 - p)**gamma * log(p)
    T term_p = pow(((T)1. - p), gamma) * log(max(p, (T)FLT_MIN));
    // p**gamma * log(1 - p)
    T term_n = pow(p, gamma) * log(max((T)1. - p, (T)FLT_MIN));

    output[index] = (T)0.;
    output[index] += -flag_p * alpha * term_p;
    output[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      output[index] *= weight[t];
    }
  }
}

template <typename T>
__global__ void sigmoid_focal_loss_backward_cuda_kernel(
    const int nthreads, const void* input_, const int64_t* target, const void* weight_,
    void* grad_input_, const T gamma, const T alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const T* input = static_cast<const T*>(input_);
    T* grad_input = static_cast<T*>(grad_input_);
    const T* weight = static_cast<const T*>(weight_);

    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + exp(-input[index]));

    // (1 - p)**gamma * (1 - p - gamma*p*log(p))
    T term_p = pow(((T)1. - p), gamma) *
               ((T)1. - p - (gamma * p * log(max(p, (T)FLT_MIN))));
    // p**gamma * (gamma * (1 - p) * log(1 - p) - p)
    T term_n = pow(p, gamma) *
               (gamma * ((T)1. - p) * log(max((T)1. - p, (T)FLT_MIN)) - p);

    grad_input[index] = (T)0.;
    grad_input[index] += -flag_p * alpha * term_p;
    grad_input[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      grad_input[index] *= weight[t];
    }
  }
}

}  // namespace cuda

}  // namespace impl

diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t input_,
                                          diopiTensorHandle_t target_,
                                          diopiTensorHandle_t weight_,
                                          diopiTensorHandle_t output_,
                                          const float gamma,
                                          const float alpha){
  auto input = impl::cuda::makeTensor(input_);
  auto target = impl::cuda::makeTensor(target_);
  auto weight = impl::cuda::makeTensor(weight_);
  auto output = impl::cuda::makeTensor(output_);

  int output_size = output.numel();
  int num_classes = input.size(1);
  // AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,"target label should smaller or equal than num classes");
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(impl::cuda::sigmoid_focal_loss_forward_cuda_kernel_diopi,
                                input.dtype(),
                                GET_BLOCKS(output_size),
                                THREADS_PER_BLOCK,
                                stream,
                                output_size,
                                input.data(),
                                static_cast<const int64_t*>(target.data()),
                                weight.data(),
                                output.data(),
                                gamma,
                                alpha,
                                num_classes);
  return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t input_,
                                          diopiTensorHandle_t target_,
                                          diopiTensorHandle_t weight_,
                                          diopiTensorHandle_t grad_input_,
                                          const float gamma,
                                          const float alpha){
  auto input = impl::cuda::makeTensor(input_);
  auto target = impl::cuda::makeTensor(target_);
  auto weight = impl::cuda::makeTensor(weight_);
  auto grad_input = impl::cuda::makeTensor(grad_input_);

  int output_size = grad_input.numel();
  int num_classes = input.size(1);
  // AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,"target label should smaller or equal than num classes");
  auto stream = impl::cuda::getStream(ctx);
  dispatch_float_types_and_half(impl::cuda::sigmoid_focal_loss_backward_cuda_kernel,
                                input.dtype(),
                                GET_BLOCKS(output_size),
                                THREADS_PER_BLOCK,
                                stream,
                                output_size,
                                input.data(),
                                static_cast<const int64_t*>(target.data()),
                                weight.data(),
                                grad_input.data(),
                                gamma,
                                alpha,
                                num_classes);
  return diopiSuccess;
}
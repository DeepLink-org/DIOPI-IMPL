#ifndef IMPL_CAMB_FUNCTIONS_ADAM_HPP_
#define IMPL_CAMB_FUNCTIONS_ADAM_HPP_

#include <cnrt.h>
#include <stdint.h>

void bang_fused_adam_internal(void* grad, void* m, void* v, void* v_max, void* variable, size_t sizes, int tensor_num, float beta1, float beta2,
                              float epsilon_correction, float learning_rate_correction, int adam_mode, float decay, float decay_correction, cnrtDim3_t k_dim,
                              cnrtFunctionType_t k_type, cnrtQueue_t queue, cnrtDataType_t cnrt_type, bool amsgrad);

#endif  // IMPL_CAMB_FUNCTIONS_ADAM_HPP_

#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t
LogicScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logic_op);
DIOPI_API diopiError_t LogicInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logic_op);
DIOPI_API diopiError_t
Logic(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logic_op);
DIOPI_API diopiError_t LogicInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logic_op);

}  // extern "C"

}  // namespace camb
}  // namespace impl

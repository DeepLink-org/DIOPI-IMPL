#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "logic.h"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_GE));
}

DIOPI_API diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_GE));
}

DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_GE));
}

DIOPI_API diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_GE));
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

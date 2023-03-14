#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "logic.h"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_LE));
}

DIOPI_API diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_LE));
}

DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_LE));
}

DIOPI_API diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_LE));
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

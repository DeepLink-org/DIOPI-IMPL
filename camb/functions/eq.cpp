#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "logic.h"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
}

DIOPI_API diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_EQ));
}

DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
}

DIOPI_API diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_EQ));
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "logic.h"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t
LogicScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto out_tensor = makeTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t other_t;
    diopiSize_t input_shape;
    DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
    DIOPI_CALL(diopiRequireTensor(ctx, &other_t, &input_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, other_t, other));
    auto other_t_tensor = makeTensor(other_t);
    CnnlTensorDesc other_t_desc(other_t_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_t_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logic_op,
                               input_desc.get(),
                               input_tensor.data(),
                               other_t_desc.get(),
                               other_t_tensor.data(),
                               workspace,
                               workspace_size,
                               out_desc.get(),
                               out_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t LogicInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t other_t;
    diopiSize_t input_shape;
    DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
    DIOPI_CALL(diopiRequireTensor(ctx, &other_t, &input_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, other_t, other));
    auto other_t_tensor = makeTensor(other_t);
    CnnlTensorDesc other_t_desc(other_t_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_t_desc.get(), input_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logic_op,
                               input_desc.get(),
                               input_tensor.data(),
                               other_t_desc.get(),
                               other_t_tensor.data(),
                               workspace,
                               workspace_size,
                               input_desc.get(),
                               input_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t
Logic(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto other_tensor = makeTensor(other);
    auto out_tensor = makeTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logic_op,
                               input_desc.get(),
                               input_tensor.data(),
                               other_desc.get(),
                               other_tensor.data(),
                               workspace,
                               workspace_size,
                               out_desc.get(),
                               out_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t LogicInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto other_tensor = makeTensor(other);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_desc.get(), input_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logic_op,
                               input_desc.get(),
                               input_tensor.data(),
                               other_desc.get(),
                               other_tensor.data(),
                               workspace,
                               workspace_size,
                               input_desc.get(),
                               input_tensor.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    /* Get handle and generate tensors */
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tr = makeTensor(input);
    auto output_tr = makeTensor(out);

    /* Some basic check */
    DIOPI_CHECK(input_tr.dim() == 3 || input_tr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    auto memory_format = MemoryFormat::ChannelsLast;
    auto input_channel_last = input_tr.contiguous(ctx, memory_format);
    cnnl_transpose(ctx, handle, input_tr, input_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    auto output_channel_last = output_tr.contiguous(ctx, memory_format);

    /* generate tensor desc */
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc input_desc(input_channel_last, layout);
    CnnlTensorDesc output_desc(output_channel_last, layout);

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingForward(handle,
                                              input_desc.get(),
                                              input_channel_last.data(),
                                              CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                              output_desc.get(),
                                              output_channel_last.data(),
                                              nullptr,
                                              nullptr));

    // NHWC -> NCHW
    cnnl_transpose(ctx, handle, output_channel_last, output_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input) {
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

#include <diopi/functions.h>

#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiNLLLoss(diopiContextHandle_t ctx,
                          diopiTensorHandle_t out,
                          diopiConstTensorHandle_t input,
                          diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight,
                          diopiReduction_t reduction,
                          int64_t ignore_index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(out);
    auto target_tr = DiopiTensor(target);
    auto weight_tr = DiopiTensor(weight);

    DIOPI_CHECK(input_tr.dtype() != diopi_dtype_float16, "Half is not supported currently")
    DIOPI_CHECK(input_tr.numel() != 0, "input tensor is empty")

    if (target_tr.dtype() != diopi_dtype_int32) {
        target_tr = dataTypeCast(ctx, target_tr, diopi_dtype_int32);
    }
    if (!weight_tr.defined()) {
        weight_tr = ones(ctx, {input_tr.shape()[1]}, input_tr.dtype());
    }
    DIOPI_CHECK(input_tr.is_contiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weight_tr.is_contiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(target_tr.is_contiguous(), "input tensor should be contiguous");

    auto input_contiguous = input_tr;

    auto dim = input_tr.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(target_tr.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(input_tr.shape()[0] == target_tr.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weight_tr.defined() || weight_tr.numel() == input_tr.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else if (dim == 3) {
        int64_t input_last_size = 1;
        for (int i = 2; i < input_tr.dim(); ++i) {
            input_last_size *= input_tr.shape()[i];
        }
        input_tr.reshape({input_tr.shape()[0], input_tr.shape()[1], 1, input_last_size});

        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto input_size = input_contiguous.shape();
    int C = input_size[1];
    int N = std::accumulate(input_size.begin(), input_size.end(), 1, std::multiplies<int64_t>()) / C;
    DIOPI_CHECK(N == target_tr.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(C == weight_tr.numel(), "Weight size need be equal as input C.");
    std::vector<int> output_size(input_size.begin(), input_size.end());

    cnnlNlllossAlgorithm_t reduction_mode;
    switch (reduction) {
        case 0: {
            reduction_mode = CNNL_REDUCTION_NONE;
            output_size.erase(output_size.begin() + 1);
            break;
        }
        case 1: {
            reduction_mode = CNNL_REDUCTION_MEAN;
            output_size = {1};
            break;
        }
        case 2: {
            reduction_mode = CNNL_REDUCTION_SUM;
            output_size = {1};
            break;
        }
        default:
            DIOPI_CHECK(false, "unexpected nll_loss reduciton mode");
    }
    auto total_weight_tr = requiresTensor(ctx, {1}, weight_tr.dtype());
    diopiScalar_t scalar({weight_tr.dtype(), static_cast<double>(target_tr.numel())});
    diopiFill(ctx, total_weight_tr.tensor_handle(), &scalar);

    CnnlTensorDesc input_desc;
    CnnlTensorDesc target_desc;
    CnnlTensorDesc weight_desc(weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc tw_desc(total_weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc;
    input_desc.set(input_contiguous, CNNL_LAYOUT_ARRAY, {N, C});
    target_desc.set(target_tr, CNNL_LAYOUT_ARRAY, {N});
    output_desc.set(output_tr, CNNL_LAYOUT_ARRAY, output_size);

    size_t workspace_size = 0;
    DIOPI_CHECKCNNL(cnnlGetNlllossWorkspaceSize(handle, input_desc.get(), &workspace_size));
    void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

    DIOPI_CALLCNNL(cnnlNlllossForward(handle,
                                      reduction_mode,
                                      workspace_ptr,
                                      workspace_size,
                                      input_desc.get(),
                                      input_contiguous.data(),
                                      target_desc.get(),
                                      target_tr.data(),
                                      static_cast<int>(ignore_index),
                                      weight_desc.get(),
                                      weight_tr.data(),
                                      tw_desc.get(),
                                      total_weight_tr.data(),
                                      output_desc.get(),
                                      output_tr.data()));

    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx,
                                  diopiTensorHandle_t grad_input,
                                  diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target,
                                  diopiConstTensorHandle_t weight,
                                  diopiReduction_t reduction,
                                  int64_t ignore_index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tr = DiopiTensor(input);
    auto grad_input_tr = DiopiTensor(grad_input);
    auto grad_output_tr = DiopiTensor(grad_output);
    auto target_tr = DiopiTensor(target);
    auto weight_tr = DiopiTensor(weight);

    DIOPI_CHECK(input_tr.dtype() != diopi_dtype_float16, "Half is not supported currently")
    DIOPI_CHECK(input_tr.numel() != 0, "input tensor is empty")

    if (target_tr.dtype() != diopi_dtype_int32) {
        target_tr = dataTypeCast(ctx, target_tr, diopi_dtype_int32);
    }
    if (!weight_tr.defined()) {
        weight_tr = ones(ctx, {input_tr.shape()[1]}, input_tr.dtype());
    }
    DIOPI_CHECK(input_tr.is_contiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weight_tr.is_contiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(target_tr.is_contiguous(), "input tensor should be contiguous");

    auto input_contiguous = input_tr;

    auto dim = input_tr.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(target_tr.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(input_tr.shape()[0] == target_tr.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weight_tr.defined() || weight_tr.numel() == input_tr.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else if (dim == 3) {
        int64_t input_last_size = 1;
        for (int i = 2; i < input_tr.dim(); ++i) {
            input_last_size *= input_tr.shape()[i];
        }
        input_tr.reshape({input_tr.shape()[0], input_tr.shape()[1], 1, input_last_size});

        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto input_size = input_contiguous.shape();
    int C = input_size[1];
    int N = std::accumulate(input_size.begin(), input_size.end(), 1, std::multiplies<int64_t>()) / C;
    DIOPI_CHECK(N == target_tr.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(C == weight_tr.numel(), "Weight size need be equal as input C.");

    cnnlNlllossAlgorithm_t reduction_mode;
    switch (reduction) {
        case 0:
            reduction_mode = CNNL_REDUCTION_NONE;
            break;
        case 1:
            reduction_mode = CNNL_REDUCTION_MEAN;
            break;
        case 2:
            reduction_mode = CNNL_REDUCTION_SUM;
            break;
        default:
            DIOPI_CHECK(false, "unexpected nll_loss reduciton mode");
    }

    auto grad_input_real_tr = requiresTensor(ctx, {N, C}, input_contiguous.dtype());

    auto total_weight_tr = requiresTensor(ctx, {1}, weight_tr.dtype());
    diopiScalar_t scalar({weight_tr.dtype(), static_cast<double>(target_tr.numel())});
    diopiFill(ctx, total_weight_tr.tensor_handle(), &scalar);

    CnnlTensorDesc grad_output_desc(grad_output_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc target_desc;
    CnnlTensorDesc weight_desc(weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc tw_desc(total_weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_input_desc(grad_input_real_tr, CNNL_LAYOUT_ARRAY);
    target_desc.set(target_tr, CNNL_LAYOUT_ARRAY, {N});

    DIOPI_CALLCNNL(cnnlNlllossBackward(handle,
                                       reduction_mode,
                                       grad_output_desc.get(),
                                       grad_output_tr.data(),
                                       target_desc.get(),
                                       target_tr.data(),
                                       static_cast<int>(ignore_index),
                                       weight_desc.get(),
                                       weight_tr.data(),
                                       tw_desc.get(),
                                       total_weight_tr.data(),
                                       grad_input_desc.get(),
                                       grad_input_real_tr.data()));
    if (dim > 2) {
        // NHWC -> NCHW
        grad_input_real_tr.reshape(input_contiguous.shape());
        grad_input_tr.reshape(input_contiguous.shape());
        cnnl_transpose(ctx, handle, grad_input_real_tr, grad_input_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);
    } else {
        diopiCopyInp(ctx, grad_input_real_tr.tensor_handle(), grad_input);
    }

    return diopiSuccess;
}

diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx,
                                   diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input,
                                   diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight,
                                   diopiReduction_t reduction,
                                   int64_t ignore_index,
                                   double label_smoothing) {
    auto input_tr = DiopiTensor(input);
    auto target_tr = DiopiTensor(target);

    DIOPI_CHECK(label_smoothing == 0, "param label_smoothing is not supported")
    DIOPI_CHECK(target_tr.dim() == input_tr.dim() - 1, "Probabilities for each class are not supported");

    auto log_tr = requiresTensor(ctx, input_tr.shape(), input_tr.dtype());
    diopiLogSoftmax(ctx, log_tr.tensor_handle(), input, 1, input_tr.dtype());
    diopiNLLLoss(ctx, out, log_tr.tensor_handle(), target, weight, reduction, ignore_index);
    return diopiSuccess;
}
diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t grad_input,
                                           diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t target,
                                           diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction,
                                           int64_t ignore_index,
                                           double label_smoothing) {
    auto input_tr = DiopiTensor(input);
    auto target_tr = DiopiTensor(target);
    auto grad_input_tr = DiopiTensor(grad_input);

    DIOPI_CHECK(label_smoothing == 0, "param label_smoothing is not supported")
    DIOPI_CHECK(target_tr.dim() == input_tr.dim() - 1, "Probabilities for each class are not supported");

    auto log_tr = requiresTensor(ctx, input_tr.shape(), input_tr.dtype());
    auto grad_tmp_tr = requiresTensor(ctx, grad_input_tr.shape(), grad_input_tr.dtype());

    diopiLogSoftmax(ctx, log_tr.tensor_handle(), input, 1, input_tr.dtype());
    // for nll loss backward, `input` should be logsoftmax out.
    diopiNLLLossBackward(ctx, grad_tmp_tr.tensor_handle(), grad_output, log_tr.tensor_handle(), target, weight, reduction, ignore_index);
    // for softmax backward, `output` should be logsoftmax out
    diopiLogSoftmaxBackward(ctx, grad_input, grad_tmp_tr.tensor_handle(), log_tr.tensor_handle(), 1, input_tr.dtype());
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl

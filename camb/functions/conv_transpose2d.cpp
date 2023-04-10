#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                             diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups,
                                             diopiSize_t dilation) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor = DiopiTensor(input);
    DiopiTensor weight_tensor = DiopiTensor(weight);
    DiopiTensor output_tensor = DiopiTensor(out);

    DiopiTensor input_casted = input_tensor;
    DiopiTensor weight_casted = weight_tensor;
    DiopiTensor output_casted = output_tensor;

    std::vector<DiopiTensor *> tensors{&input_casted, &weight_casted, &output_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    CnnlTensorDesc input_desc(input_casted, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc weight_desc(weight_casted, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc output_desc(output_casted, CNNL_LAYOUT_NCHW);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> conv_desc;

    std::vector<int> stride_vec{stride.data, stride.data + stride.len};
    std::vector<int> padding_vec{padding.data, padding.data + padding.len};
    std::vector<int> dilation_vec{dilation.data, dilation.data + dilation.len};

    int padding_[4] = {padding_vec[0], padding_vec[1], padding_vec[0], padding_vec[1]};
    int stride_[2] = {stride_vec[0], stride_vec[1]};
    int dilation_[2] = {dilation_vec[0], dilation_vec[1]};

    cnnlDataType_t compute_type;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&compute_type, input_casted.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(conv_desc.get(), 4, padding_, stride_, dilation_, groups, compute_type));

    size_t workspace_size_input;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardDataWorkspaceSize(
        handle, weight_desc.get(), input_desc.get(), conv_desc.get(), output_desc.get(), CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT, &workspace_size_input));
    void *workspace_input;
    if (workspace_size_input != 0) {
        workspace_input = requiresBuffer(ctx, workspace_size_input).data();
    }
    float alpha = 1.0;
    float beta = 1.0;
    cnnlConvolutionBackwardData(handle,
                                &alpha,
                                weight_desc.get(),
                                weight_casted.data(),
                                input_desc.get(),
                                input_casted.data(),
                                conv_desc.get(),
                                CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                workspace_input,
                                workspace_size_input,
                                &beta,
                                output_desc.get(),
                                output_casted.data());

    DiopiTensor bias_tensor = DiopiTensor(bias);
    if (bias_tensor.defined()) {
        DiopiTensor bias_casted = bias_tensor;

        std::vector<DiopiTensor *> tensors{&bias_casted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        CnnlTensorDesc bias_desc(bias_casted, CNNL_LAYOUT_ARRAY);

        DiopiTensor output_t;
        std::vector<int64_t> axis{0, 2, 3, 1};
        std::vector<int64_t> src_shape_t_64(output_casted.shape().size());
        for (int i = 0; i < output_casted.shape().size(); ++i) {
            src_shape_t_64[i] = output_casted.shape()[axis[i]];
        }
        diopiSize_t output_t_shape(src_shape_t_64.data(), src_shape_t_64.size());
        auto output_t_handle = output_t.tensorHandle();
        DIOPI_CALL(diopiRequireTensor(ctx, &output_t_handle, &output_t_shape, nullptr, output_casted.dtype(), diopi_device));
        diopiSize_t nchw2nhwc(axis.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, output_t_handle, output_casted.tensorHandle(), nchw2nhwc));
        output_t = DiopiTensor(output_t_handle);
        CnnlTensorDesc output_t_desc(output_t, CNNL_LAYOUT_ARRAY);

        size_t workspace_size_bias;
        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, bias_desc.get(), output_t_desc.get(), &workspace_size_bias));

        void *workspace_bias = nullptr;
        if (workspace_size_bias != 0) {
            workspace_bias = requiresBuffer(ctx, workspace_size_bias).data();
        }

        DIOPI_CALLCNNL(
            cnnlBiasAdd(handle, &alpha, bias_desc.get(), bias_casted.data(), workspace_bias, workspace_size_bias, &beta, output_t_desc.get(), output_t.data()));

        std::vector<int64_t> perm_nhwc2nchw{0, 3, 1, 2};
        diopiSize_t nhwc2nchw(perm_nhwc2nchw.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, output_casted.tensorHandle(), output_t.tensorHandle(), nhwc2nchw));
    }
    DIOPI_CALL(dataTypeCast(ctx, output_tensor, output_casted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

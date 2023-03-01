#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

extern "C" {

static diopiError_t diopiTransposeNCHW2NHWC(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out) {
    auto stream = impl::camb::getStream(ctx);
    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> inputDesc, outDesc;
    cnnlTensorDescriptor_t input_desc = inputDesc.get();
    cnnlTensorDescriptor_t out_desc = outDesc.get();

    cnnlDataType_t cnnl_dtype;
    diopiDtype_t diopi_dtype;
    diopiGetTensorDtype(input, &diopi_dtype);
    convertType(&cnnl_dtype, diopi_dtype);

    diopiSize_t input_shape;
    diopiGetTensorShape(input, &input_shape);
    diopiSize_t inputStride;
    diopiGetTensorStride(input, &inputStride);
    std::vector<int> input_dims(input_shape.len), out_dims(input_shape.len);
    std::vector<int> input_stride(input_shape.len);
    for (int i = 0; i < input_shape.len; i++) {
        input_dims[i] = input_shape.data[i];
        out_dims[i] = input_shape.data[i];
        input_stride[i] = inputStride.data[i];
    }
    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(input_desc, CNNL_LAYOUT_NCHW, cnnl_dtype, input_shape.len, input_dims.data(), input_stride.data()));
    out_dims[1] = input_dims[2];
    out_dims[2] = input_dims[3];
    out_dims[3] = input_dims[1];
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(out_desc, CNNL_LAYOUT_NHWC, cnnl_dtype, input_shape.len, out_dims.data()));

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> transDesc;
    cnnlTransposeDescriptor_t trans_desc = transDesc.get();
    std::vector<int> order = {0, 2, 3, 1};
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(trans_desc, input_shape.len, order.data()));

    const void* input_data = nullptr;
    void* out_data = nullptr;
    diopiGetTensorDataConst(&input, &input_data);
    diopiGetTensorData(&out, &out_data);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, input_desc, trans_desc, &workspace_size));
    auto workspace = impl::camb::requiresBuffer(ctx, workspace_size);

    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, trans_desc, input_desc, input_data, out_desc, out_data, workspace.data(), workspace_size));
    return diopiSuccess;
}

static diopiError_t diopiTransposeNHWC2NCHW(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out) {
    auto stream = impl::camb::getStream(ctx);
    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> inputDesc, outDesc;
    cnnlTensorDescriptor_t input_desc = inputDesc.get();
    cnnlTensorDescriptor_t out_desc = outDesc.get();

    cnnlDataType_t cnnl_dtype;
    diopiDtype_t diopi_dtype;
    diopiGetTensorDtype(input, &diopi_dtype);
    convertType(&cnnl_dtype, diopi_dtype);

    diopiSize_t input_shape;
    diopiGetTensorShape(input, &input_shape);
    diopiSize_t inputStride;
    diopiGetTensorStride(input, &inputStride);
    std::vector<int> input_dims(input_shape.len), out_dims(input_shape.len);
    std::vector<int> input_stride(input_shape.len);
    for (int i = 0; i < input_shape.len; i++) {
        input_dims[i] = input_shape.data[i];
        out_dims[i] = input_shape.data[i];
        input_stride[i] = inputStride.data[i];
    }
    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(input_desc, CNNL_LAYOUT_NHWC, cnnl_dtype, input_shape.len, input_dims.data(), input_stride.data()));
    out_dims[1] = input_dims[3];
    out_dims[2] = input_dims[1];
    out_dims[3] = input_dims[2];
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(out_desc, CNNL_LAYOUT_NCHW, cnnl_dtype, input_shape.len, out_dims.data()));

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> transDesc;
    cnnlTransposeDescriptor_t trans_desc = transDesc.get();
    std::vector<int> order = {0, 3, 1, 2};
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(trans_desc, input_shape.len, order.data()));

    const void* input_data = nullptr;
    void* out_data = nullptr;
    diopiGetTensorDataConst(&input, &input_data);
    diopiGetTensorData(&out, &out_data);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, input_desc, trans_desc, &workspace_size));
    auto workspace = impl::camb::requiresBuffer(ctx, workspace_size);

    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, trans_desc, input_desc, input_data, out_desc, out_data, workspace.data(), workspace_size));
    return diopiSuccess;
}

static diopiTensorHandle_t diopiRequireNCHW2NHWC(diopiContextHandle_t ctx, diopiConstTensorHandle_t input) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiSize_t nchw_shape;
    diopiGetTensorShape(input, &nchw_shape);
    std::vector<int64_t> shape(4);
    shape[0] = nchw_shape.data[0];
    shape[1] = nchw_shape.data[2];
    shape[2] = nchw_shape.data[3];
    shape[3] = nchw_shape.data[1];
    diopiSize_t nhwc_shape(shape.data(), 4);
    diopiTensorHandle_t out = nullptr;
    diopiRequireTensor(ctx, &out, &nhwc_shape, nullptr, dtype, diopi_device);
    diopiTransposeNCHW2NHWC(ctx, input, out);
    return out;
}

static diopiError_t diopiPrepareConv2dDesc(
    cnnlConvolutionDescriptor_t desc, diopiConstTensorHandle_t input, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    std::vector<int> conv_padding(4);
    if (4 == padding.len) {
        for (int i = 0; i < 4; i++) {
            conv_padding[i] = padding.data[i];
        }
    } else if (2 == padding.len) {
        conv_padding[0] = conv_padding[1] = padding.data[0];
        conv_padding[2] = conv_padding[3] = padding.data[1];
    } else if (1 == padding.len) {
        for (int i = 0; i < 4; i++) {
            conv_padding[i] = padding.data[0];
        }
    } else {
        return diopiErrorOccurred;
    }

    std::vector<int> conv_stride(stride.len), conv_dilation(dilation.len);
    for (int i = 0; i < stride.len; i++) {
        conv_stride[i] = stride.data[i];
    }
    for (int i = 0; i < dilation.len; i++) {
        conv_dilation[i] = dilation.data[i];
    }

    diopiSize_t input_shape;
    diopiGetTensorShape(input, &input_shape);
    cnnlDataType_t cnnl_dtype;
    diopiDtype_t diopi_dtype;
    diopiGetTensorDtype(input, &diopi_dtype);
    DIOPI_CALL(convertType(&cnnl_dtype, diopi_dtype));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(desc, input_shape.len, conv_padding.data(), conv_stride.data(), conv_dilation.data(), groups, cnnl_dtype));
    return diopiSuccess;
}

static diopiTensorHandle_t diopiRequireNCHWConv2dOut(diopiContextHandle_t ctx,
                                                     cnnlConvolutionDescriptor_t conv_desc,
                                                     diopiConstTensorHandle_t input,
                                                     diopiConstTensorHandle_t weight) {
    auto at_input = impl::camb::makeTensor(input);
    auto at_weight = impl::camb::makeTensor(weight);
    CnnlTensorDesc input_desc(at_input, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weight_desc(at_weight, CNNL_LAYOUT_NHWC);

    std::vector<int> out_shape(4);
    std::vector<int64_t> out_shape64(4);
    cnnlGetConvolutionForwardOutputDim(conv_desc, input_desc.get(), weight_desc.get(), 4, out_shape.data());
    for (int i = 0; i < 4; i++) {
        out_shape64[i] = out_shape[i];
    }
    diopiSize_t shape(out_shape64.data(), 4);

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiTensorHandle_t out = nullptr;
    diopiRequireTensor(ctx, &out, &shape, nullptr, dtype, diopi_device);
    return out;
}

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx,
                                diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias,
                                diopiSize_t stride,
                                diopiSize_t padding,
                                diopiSize_t dilation,
                                int64_t groups) {
    auto stream = impl::camb::getStream(ctx);
    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;
    cnnlConvolutionDescriptor_t conv_desc = convDesc.get();
    diopiPrepareConv2dDesc(conv_desc, input, stride, padding, dilation, groups);

    diopiConstTensorHandle_t nhwc_input = diopiRequireNCHW2NHWC(ctx, input);
    diopiConstTensorHandle_t nhwc_weight = diopiRequireNCHW2NHWC(ctx, weight);
    diopiTensorHandle_t nhwc_out = diopiRequireNCHWConv2dOut(ctx, conv_desc, nhwc_input, nhwc_weight);
    auto at_nhwc_input = impl::camb::makeTensor(nhwc_input);
    auto at_nhwc_weight = impl::camb::makeTensor(nhwc_weight);
    auto at_nhwc_out = impl::camb::makeTensor(nhwc_out);

    CnnlTensorDesc input_desc(at_nhwc_input, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weight_desc(at_nhwc_weight, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc out_desc(at_nhwc_out, CNNL_LAYOUT_NHWC);
    cnnlTensorDescriptor_t bias_descriptor = nullptr;
    CnnlTensorDesc bias_desc;
    if (nullptr != bias) {
        auto at_nhwc_bias = impl::camb::makeTensor(bias);
        bias_desc.set(at_nhwc_bias, CNNL_LAYOUT_ARRAY);
        bias_descriptor = bias_desc.get();
    }

    cnnlConvolutionForwardAlgo_t algo;
    DIOPI_CALLCNNL(
        cnnlGetConvolutionForwardAlgorithm(handle, conv_desc, input_desc.get(), weight_desc.get(), out_desc.get(), CNNL_CONVOLUTION_FWD_FASTEST, &algo));

    size_t workspace_size(0);
    DIOPI_CALLCNNL(
        cnnlGetConvolutionForwardWorkspaceSize(handle, input_desc.get(), weight_desc.get(), out_desc.get(), bias_descriptor, conv_desc, algo, &workspace_size));
    auto workspace = impl::camb::requiresBuffer(ctx, workspace_size);

    const void* input_data = nullptr;
    diopiGetTensorDataConst(&nhwc_input, &input_data);
    const void* weight_data = nullptr;
    diopiGetTensorDataConst(&nhwc_weight, &weight_data);
    const void* bias_data = nullptr;
    if (nullptr != bias) {
        diopiGetTensorDataConst(&bias, &bias_data);
    }
    void* out_data = nullptr;
    diopiGetTensorData(&nhwc_out, &out_data);

    DIOPI_CALLCNNL(cnnlConvolutionForward(handle,
                                          conv_desc,
                                          algo,
                                          nullptr,
                                          input_desc.get(),
                                          input_data,
                                          weight_desc.get(),
                                          weight_data,
                                          bias_descriptor,
                                          bias_data,
                                          workspace.data(),
                                          workspace_size,
                                          nullptr,
                                          out_desc.get(),
                                          out_data));
    diopiTransposeNHWC2NCHW(ctx, nhwc_out, out);
    return diopiSuccess;
}

}  // extern "C"

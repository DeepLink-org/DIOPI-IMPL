/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions.h>
#include <stdio.h>
#include <dlfcn.h>

static void* handle;

static void
__attribute__ ((constructor))
diopi_init(void) {
    handle = dlopen("libdiopi_real_impl.so", RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
    printf("diopi dyload init\n");
    if (!handle) {
        fprintf (stderr, "%s ", dlerror());
    }
}

static void
__attribute__ ((destructor))
diopi_fini(void)
{
dlclose(handle);
}

DIOPI_RT_API const char* diopiGetVendorName() {
    const char* (*func)();
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGetVendorName"));
    if (func != NULL)
        return (*func)();
    else
        return "diopiGetVendorName not implemented!";
}

DIOPI_RT_API const char* diopiGetImplVersion() {
    const char* (*func)();
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGetImplVersion"));
    if (func != NULL)
        return (*func)();
    else
        return "diopiGetImplVersion not implemented!";
}

DIOPI_RT_API const char* diopiGetLastErrorString() {
    const char* (*func)();
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGetLastErrorString"));
    if (func != NULL)
        return (*func)();
    else
        return "diopiGetLastErrorString not implemented!";
}

DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                          diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        diopiSize_t, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution2d"));
    if (func != NULL)
        return (*func)(ctx, out, input, weight, bias, stride, padding, dilation, groups);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t *, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution2dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_weight, grad3, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                      diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                                      diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, bool, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBatchNorm"));
    if (func != NULL)
        return (*func)(ctx, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                              diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                              diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, bool, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBatchNormBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, training, eps);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRelu"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiReluInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardtanh"));
    if (func != NULL)
        return (*func)(ctx, out, input, min_val, max_val);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardtanhInp"));
    if (func != NULL)
        return (*func)(ctx, input, min_val, max_val);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardtanhBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, min_val, max_val);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardswish"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardswishInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardswishBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     const diopiScalar_t* threshold, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThreshold"));
    if (func != NULL)
        return (*func)(ctx, out, input, threshold, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThresholdInp"));
    if (func != NULL)
        return (*func)(ctx, input, threshold, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThresholdBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, threshold);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, const char* approximate) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGelu"));
    if (func != NULL)
        return (*func)(ctx, out, input, approximate);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input, const char* approximate) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGeluBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, approximate);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeakyRelu"));
    if (func != NULL)
        return (*func)(ctx, out, input, negative_slope);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeakyReluInp"));
    if (func != NULL)
        return (*func)(ctx, input, negative_slope);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeakyReluBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, negative_slope, input_is_result);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                      bool count_include_pad, const int64_t* divisor_override) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, bool,
        bool, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAvgPool2d"));
    if (func != NULL)
        return (*func)(ctx, out, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                              bool count_include_pad, const int64_t* divisor_override) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, bool,
        bool, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAvgPool2dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool2d"));
    if (func != NULL)
        return (*func)(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                 diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                                                 diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t,
        diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool2dWithIndices"));
    if (func != NULL)
        return (*func)(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool2dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool2d"));
    if (func != NULL)
        return (*func)(ctx, out, input, output_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                      diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool2dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool2d"));
    if (func != NULL)
        return (*func)(ctx, out, input, output_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                         diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool2dWithIndices"));
    if (func != NULL)
        return (*func)(ctx, out, indices, input, output_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool2dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, indices);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask,
                                    diopiConstTensorHandle_t input, double p, bool train) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDropout"));
    if (func != NULL)
        return (*func)(ctx, out, mask, input, p, train);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask,
                                       double p, bool train) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDropoutInp"));
    if (func != NULL)
        return (*func)(ctx, input, mask, p, train);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMSELoss"));
    if (func != NULL)
        return (*func)(ctx, out, input, target, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMSELossBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, target, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs,
                                             diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, float, float, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidFocalLoss"));
    if (func != NULL)
        return (*func)(ctx, out, inputs, targets, alpha, gamma, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                                     diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiTensorHandle_t, float, float, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidFocalLossBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_output, input, target, grad_input, gamma, alpha, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                             diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                             int64_t ignore_index, double label_smoothing) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t,
        int64_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCrossEntropyLoss"));
    if (func != NULL)
        return (*func)(ctx, out, input, target, weight, reduction, ignore_index, label_smoothing);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiReduction_t, int64_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCrossEntropyLossBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index, label_smoothing);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                    int64_t ignore_index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNLLLoss"));
    if (func != NULL)
        return (*func)(ctx, out, input, target, weight, reduction, ignore_index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignore_index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiReduction_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNLLLossBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCEWithLogits"));
    if (func != NULL)
        return (*func)(ctx, out, input, target, weight, pos_weight, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCEWithLogitsBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, target, weight, pos_weight, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCELoss"));
    if (func != NULL)
        return (*func)(ctx, out, input, target, weight, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCELossBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, target, weight, reduction);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSign"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAbsInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAbs"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNegInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNeg"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFloorInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFloor"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSqrtInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSqrt"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRsqrtInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRsqrt"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSinInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSin"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCosInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCos"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTanhInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTanh"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                         diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTanhBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, output);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoid"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, output);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSiluInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSilu"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSiluBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiExpInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiExp"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog2Inp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog2"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog10Inp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog10"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiErfInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiErf"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPowScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, exponent);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPow"));
    if (func != NULL)
        return (*func)(ctx, out, input, exponent);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPowInp"));
    if (func != NULL)
        return (*func)(ctx, input, exponent);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPowTensor"));
    if (func != NULL)
        return (*func)(ctx, out, input, exponent);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPowInpTensor"));
    if (func != NULL)
        return (*func)(ctx, input, exponent);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdd"));
    if (func != NULL)
        return (*func)(ctx, out, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                   diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddInp"));
    if (func != NULL)
        return (*func)(ctx, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                         const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSub"));
    if (func != NULL)
        return (*func)(ctx, out, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                   diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSubInp"));
    if (func != NULL)
        return (*func)(ctx, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSubScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                         const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSubInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMul"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMulInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMulScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMulInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiRoundMode_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDiv"));
    if (func != NULL)
        return (*func)(ctx, out, input, other, rounding_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiRoundMode_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDivInp"));
    if (func != NULL)
        return (*func)(ctx, input, other, rounding_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, diopiRoundMode_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDivScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other, rounding_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                         const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        const diopiScalar_t*, diopiRoundMode_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDivInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other, rounding_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBmm"));
    if (func != NULL)
        return (*func)(ctx, out, input, mat2);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBaddbmm"));
    if (func != NULL)
        return (*func)(ctx, out, input, batch1, batch2, beta, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
                                       diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBaddbmmInp"));
    if (func != NULL)
        return (*func)(ctx, input, batch1, batch2, beta, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddcmul"));
    if (func != NULL)
        return (*func)(ctx, out, input, tensor1, tensor2, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddcmulInp"));
    if (func != NULL)
        return (*func)(ctx, input, tensor1, tensor2, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMatmul"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddcdiv"));
    if (func != NULL)
        return (*func)(ctx, out, input, tensor1, tensor2, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddcdivInp"));
    if (func != NULL)
        return (*func)(ctx, input, tensor1, tensor2, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t mat1, diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddmm"));
    if (func != NULL)
        return (*func)(ctx, out, input, mat1, mat2, beta, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCholesky(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper, bool checkerror) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCholesky"));
    if (func != NULL)
        return (*func)(ctx, out, info, mat, upper, checkerror);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCholeskyBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t L, bool upper) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCholeskyBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_mat, grad_output, L, upper);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTriangularSolve(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b,
                                            diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, bool, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTriangularSolve"));
    if (func != NULL)
        return (*func)(ctx, out, cloned_mat, b, mat, upper, transpose, unitriangular);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTriangularSolveBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat,
                                                    diopiConstTensorHandle_t x, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, bool, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTriangularSolveBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_b, grad_mat, grad_x, grad_cloned_mat, x, b, mat, upper, transpose, unitriangular);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, min, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampInp"));
    if (func != NULL)
        return (*func)(ctx, input, min, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, min, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClamp"));
    if (func != NULL)
        return (*func)(ctx, out, input, min, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMaxInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMaxInp"));
    if (func != NULL)
        return (*func)(ctx, input, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMaxScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMax"));
    if (func != NULL)
        return (*func)(ctx, out, input, max);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMinInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, min);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMinInp"));
    if (func != NULL)
        return (*func)(ctx, input, min);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMinScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, min);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMin"));
    if (func != NULL)
        return (*func)(ctx, out, input, min);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFill"));
    if (func != NULL)
        return (*func)(ctx, input, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogicalAnd"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogicalAndInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogicalOr"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogicalOrInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogicalNot"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogicalNotInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseAnd"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseAndInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseAndScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseAndInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseOr"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseOrInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseOrScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseOrInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseNot"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseNotInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEqScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEqInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEq"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEqInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNeScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNeInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNe"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNeInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGeScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGeInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGe"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGeInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGtScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGtInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGt"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGtInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLe"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLtScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLtInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLt"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLtInp"));
    if (func != NULL)
        return (*func)(ctx, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMean"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSum"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiStd"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, unbiased);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices,
                                diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMin"));
    if (func != NULL)
        return (*func)(ctx, min, min_indices, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMinAll"));
    if (func != NULL)
        return (*func)(ctx, min, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices,
                                diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMax"));
    if (func != NULL)
        return (*func)(ctx, max, max_indices, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxAll"));
    if (func != NULL)
        return (*func)(ctx, max, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAny"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAll"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSoftmax"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSoftmaxBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, output, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogSoftmax"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t output, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogSoftmaxBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, output, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t* indices, int64_t nums) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t*, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndex"));
    if (func != NULL)
        return (*func)(ctx, out, input, indices, nums);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
                                          diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t*, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, zeros_like_input, indices, nums, grad);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                        diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexSelect"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad,
                                                diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexSelectBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad, input_sizes, dim, index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSelect(diopiContextHandle_t ctx,  diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    diopiError_t (*func)(diopiContextHandle_t,  diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSelect"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                           diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSelectBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input_sizes, dim, index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t src, int64_t dim, int64_t index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSelectScatter"));
    if (func != NULL)
        return (*func)(ctx, out, input, src, dim, index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                         diopiConstTensorHandle_t src, int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSliceScatter"));
    if (func != NULL)
        return (*func)(ctx, out, input, src, dim, start, end, step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input,
                                  int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSlice"));
    if (func != NULL)
        return (*func)(ctx, null_out, input, dim, start, end, step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, int64_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSliceBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input_sizes, dim, start, end, step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedScatter"));
    if (func != NULL)
        return (*func)(ctx, out, input, mask, source);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                diopiConstTensorHandle_t scores, double iou_threshold) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNms"));
    if (func != NULL)
        return (*func)(ctx, out, dets, scores, iou_threshold);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNonzero"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLinear"));
    if (func != NULL)
        return (*func)(ctx, out, input, weight, bias);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                           diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLinearBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height,
                                     int64_t pooled_width, int64_t sampling_ratio, bool aligned) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, double, int64_t,
        int64_t, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlign"));
    if (func != NULL)
        return (*func)(ctx, out, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                             diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height,
                                             int64_t pooled_width, int64_t batch_size, int64_t channels, int64_t height,
                                             int64_t width, int64_t sampling_ratio, bool aligned) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, double, int64_t,
        int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlignBackward"));
    if (func != NULL)
        return (*func)(ctx, out, grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf,
                                double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        double, double, double, double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSgd"));
    if (func != NULL)
        return (*func)(ctx, w, dw, buf, lr, momentum, dampening, weight_decay, nesterov);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t *grads,
                                         int64_t num_grads, double max_norm, double norm_type, bool error_if_nonfinite) {
    diopiError_t (*func)(diopiContextHandle_t, double*, diopiTensorHandle_t *,
        int64_t, double, double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClipGradNorm"));
    if (func != NULL)
        return (*func)(ctx, out, grads, num_grads, max_norm, norm_type, error_if_nonfinite);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout,
                                             diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEmbeddingRenorm_"));
    if (func != NULL)
        return (*func)(ctx, inout, indices, max_norm, norm_type);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t indices, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEmbedding"));
    if (func != NULL)
        return (*func)(ctx, out, weight, indices, padding_idx, scale_grad_byfreq, sparse);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                              diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEmbeddingBackward"));
    if (func != NULL)
        return (*func)(ctx, out, grad, indices, num_weights, padding_idx, scale_grad_byfreq, sparse);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTril"));
    if (func != NULL)
        return (*func)(ctx, out, input, diagonal);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t*, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCat"));
    if (func != NULL)
        return (*func)(ctx, out, tensors, num_inputs, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs,
                                           diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t*, int64_t,
        diopiConstTensorHandle_t, const diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSplitWithSizes"));
    if (func != NULL)
        return (*func)(ctx, outs, num_outs, input, splitSizes, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                  diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t*, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiStack"));
    if (func != NULL)
        return (*func)(ctx, out, tensors, numTensors, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, bool, const bool*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSort"));
    if (func != NULL)
        return (*func)(ctx, values, indices, input, dim, descending, stable);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTopk"));
    if (func != NULL)
        return (*func)(ctx, values, indices, input, k, dim, largest, sorted);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTranspose"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim0, dim1);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input, int64_t num_classes) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiOneHot"));
    if (func != NULL)
        return (*func)(ctx, out, input, num_classes);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiWhere"));
    if (func != NULL)
        return (*func)(ctx, out, condition, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                       diopiConstTensorHandle_t value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFill"));
    if (func != NULL)
        return (*func)(ctx, out, input, mask, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFillInp"));
    if (func != NULL)
        return (*func)(ctx, input, mask, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                             const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFillScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, mask, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFillInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, mask, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiReciprocal"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiReciprocalInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                                  diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq,
                                  float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        float, float, float, float, float, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdamW"));
    if (func != NULL)
        return (*func)(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                            diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        diopiSize_t, diopiSize_t, int64_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvTranspose2d"));
    if (func != NULL)
        return (*func)(ctx, out, input, weight, bias, stride, padding, output_padding, groups, dilation);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUnfold"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, size, step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUnfoldBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input_sizes, dim, size, step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCumsum"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2,
                                  double p, const int64_t* compute_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        double, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCdist"));
    if (func != NULL)
        return (*func)(ctx, out, input1, input2, p, compute_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, double, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCdistBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input1, input2, p, cdist);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const int64_t*, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiArgmax"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, keepdim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                     diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, float, float, float, float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdadelta"));
    if (func != NULL)
        return (*func)(ctx, input, grad, square_avg, acc_delta, lr, rho, eps, weight_decay);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq,
                                 diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, float, float, float, float, float, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdam"));
    if (func != NULL)
        return (*func)(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRmsprop(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t grad_avg,
                                    diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, float, float, float, float, float, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRmsprop"));
    if (func != NULL)
        return (*func)(ctx, input, grad, square_avg, grad_avg, momentum_buf, lr, alpha, eps, weight_decay, momentum, centered);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                         diopiReduction_t reduction, double beta) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiReduction_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSmoothL1Loss"));
    if (func != NULL)
        return (*func)(ctx, out, input, target, reduction, beta);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSmoothL1LossBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, target, reduction, beta);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                          diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        diopiSize_t, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution3d"));
    if (func != NULL)
        return (*func)(ctx, out, input, weight, bias, stride, padding, dilation, groups);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t *, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution3dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool3d"));
    if (func != NULL)
        return (*func)(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                 diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                                                 diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t,
        diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool3dWithIndices"));
    if (func != NULL)
        return (*func)(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool3dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool3d"));
    if (func != NULL)
        return (*func)(ctx, out, input, output_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                      diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool3dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool3d"));
    if (func != NULL)
        return (*func)(ctx, out, input, output_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool3dWithIndices"));
    if (func != NULL)
        return (*func)(ctx, out, indices, input, output_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool3dBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, indices);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedSelect"));
    if (func != NULL)
        return (*func)(ctx, out, input, mask);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                 diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedSelectBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, mask);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaximum"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMinimum"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMm"));
    if (func != NULL)
        return (*func)(ctx, out, input, mat2);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                            int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFillScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, index, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFill"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, index, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                               int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFillInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, dim, index, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                         int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFillInp"));
    if (func != NULL)
        return (*func)(ctx, input, dim, index, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiExpand"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLinspace"));
    if (func != NULL)
        return (*func)(ctx, out, start, end, steps);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPermute"));
    if (func != NULL)
        return (*func)(ctx, out, input, dims);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode, double* value) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t, const char*, double*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPad"));
    if (func != NULL)
        return (*func)(ctx, out, input, pad, mode, value);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoll"));
    if (func != NULL)
        return (*func)(ctx, out, input, shifts, dims);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFlip"));
    if (func != NULL)
        return (*func)(ctx, out, input, dims);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNorm"));
    if (func != NULL)
        return (*func)(ctx, out, input, p, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, int64_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGroupNorm"));
    if (func != NULL)
        return (*func)(ctx, out, save_mean, save_invstd, input, weight, bias, num_groups, eps);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean,
                                              diopiConstTensorHandle_t rstd, int64_t num_groups) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGroupNormBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, mean, rstd, num_groups);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim,
                                   bool sorted, bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t, const int64_t*,
        bool, bool, diopiTensorHandle_t, diopiTensorHandle_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUnique"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, sorted, return_counts, indices, counts);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiProd"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha,
                                    diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                    diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiReduction_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCTCLoss"));
    if (func != NULL)
        return (*func)(ctx, out, neg_log_likelihood, log_alpha, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets,
                                            diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha,
                                            int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiReduction_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCTCLossBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, reduction, zero_infinity);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRemainderTensor"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRemainderScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRemainder"));
    if (func != NULL)
        return (*func)(ctx, out, input, other);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGather"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGatherBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, input, dim, index);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatterInp"));
    if (func != NULL)
        return (*func)(ctx, input, dim, src, index, reduce);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, int64_t, const diopiScalar_t*, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatterInpScalar"));
    if (func != NULL)
        return (*func)(ctx, input, dim, value, index, reduce);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatter"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, src, index, reduce);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, const diopiScalar_t*, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatterScalar"));
    if (func != NULL)
        return (*func)(ctx, out, input, dim, value, index, reduce);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t*, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexPutInp"));
    if (func != NULL)
        return (*func)(ctx, input, values, indices, indices_counts, accumulate);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t*, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexPut"));
    if (func != NULL)
        return (*func)(ctx, out, input, values, indices, indices_counts, accumulate);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, int64_t, const int64_t*, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRandomInp"));
    if (func != NULL)
        return (*func)(ctx, inout, from, to, idx);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, double, double, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUniformInp"));
    if (func != NULL)
        return (*func)(ctx, inout, from, to, idx);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBernoulli"));
    if (func != NULL)
        return (*func)(ctx, out, input, idx);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBernoulliInp"));
    if (func != NULL)
        return (*func)(ctx, inout, idx);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, double, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBernoulliScalar"));
    if (func != NULL)
        return (*func)(ctx, out, p, idx);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiArange"));
    if (func != NULL)
        return (*func)(ctx, out, start, end, step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRandperm"));
    if (func != NULL)
        return (*func)(ctx, out, n, idx);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNormal"));
    if (func != NULL)
        return (*func)(ctx, out, mean, std);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNormalTensorScalar"));
    if (func != NULL)
        return (*func)(ctx, out, mean, std);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, double, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNormalScalarTensor"));
    if (func != NULL)
        return (*func)(ctx, out, mean, std);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNormalTensor"));
    if (func != NULL)
        return (*func)(ctx, out, mean, std);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNormalInp"));
    if (func != NULL)
        return (*func)(ctx, inout, mean, std);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t*, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMeshGrid"));
    if (func != NULL)
        return (*func)(ctx, outs, inputs, inputsNum);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMultinomial"));
    if (func != NULL)
        return (*func)(ctx, out, input, num_samples, replacement);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiSize_t normalized_shape, double eps) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLayerNorm"));
    if (func != NULL)
        return (*func)(ctx, out, save_mean, save_invstd, input, weight, bias, normalized_shape, eps);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                              diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLayerNormBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias, mean, rstd, normalized_shape);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiConstTensorHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCopyInp"));
    if (func != NULL)
        return (*func)(ctx, src, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleNearest"));
    if (func != NULL)
        return (*func)(ctx, out, input, size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiSize_t out_size, diopiSize_t in_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleNearestBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, out_size, in_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
                                           bool align_corners, const char* mode) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        bool, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleLinear"));
    if (func != NULL)
        return (*func)(ctx, out, input, size, align_corners, mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx,  diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                   diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode) {
    diopiError_t (*func)(diopiContextHandle_t,  diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, bool, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleLinearBackward"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, out_size, in_size, align_corners, mode);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiErfinv"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiErfinvInp"));
    if (func != NULL)
        return (*func)(ctx, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIm2Col"));
    if (func != NULL)
        return (*func)(ctx, out, input, kernel_size, dilation, padding, stride);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiSize_t output_size, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, diopiSize_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCol2Im"));
    if (func != NULL)
        return (*func)(ctx, out, input, output_size, kernel_size, dilation, padding, stride);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRepeat"));
    if (func != NULL)
        return (*func)(ctx, out, input, repeats_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCastDtype"));
    if (func != NULL)
        return (*func)(ctx, out, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAssignScoreWithkMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t output,
                                                 diopiConstTensorHandle_t points,
                                                 diopiConstTensorHandle_t centers,
                                                 diopiConstTensorHandle_t scores,
                                                 diopiConstTensorHandle_t knn_idx,
                                                 int64_t B,
                                                 int64_t N,
                                                 int64_t npoint,
                                                 int64_t M,
                                                 int64_t K,
                                                 int64_t out_dim,
                                                 int64_t aggregate) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAssignScoreWithkMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, points, centers, scores, knn_idx, B, N, npoint, M, K, out_dim, aggregate);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiAssignScoreWithkBackwardMmcv(diopiContextHandle_t ctx,
                                                         diopiTensorHandle_t grad_points,
                                                         diopiTensorHandle_t grad_centers,
                                                         diopiTensorHandle_t grad_scores,
                                                         diopiConstTensorHandle_t grad_out,
                                                         diopiConstTensorHandle_t points,
                                                         diopiConstTensorHandle_t centers,
                                                         diopiConstTensorHandle_t scores,
                                                         diopiConstTensorHandle_t knn_idx,
                                                         int64_t B,
                                                         int64_t N,
                                                         int64_t npoint,
                                                         int64_t M,
                                                         int64_t K,
                                                         int64_t out_dim,
                                                         int64_t aggregate) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAssignScoreWithkBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_points, grad_centers, grad_scores, grad_out, points, centers, scores, knn_idx, B, N, npoint, M, K, out_dim, aggregate);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBallQueryMmcv(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t idx,
                                          diopiConstTensorHandle_t center_xyz,
                                          diopiConstTensorHandle_t xyz,
                                          int64_t B,
                                          int64_t N,
                                          int64_t npoint,
                                          int64_t sample_num,
                                          float min_radius,
                                          float max_radius) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBallQueryMmcv"));
    if (func != NULL)
        return (*func)(ctx, idx, center_xyz, xyz, B, N, npoint, sample_num, min_radius, max_radius);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiStackBallQueryMmcv(diopiContextHandle_t ctx,
                                               diopiTensorHandle_t idx,
                                               diopiConstTensorHandle_t center_xyz,
                                               diopiConstTensorHandle_t center_xyz_batch_cnt,
                                               diopiConstTensorHandle_t xyz,
                                               diopiConstTensorHandle_t xyz_batch_cnt,
                                               float max_radius,
                                               int64_t sample_num) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiStackBallQueryMmcv"));
    if (func != NULL)
        return (*func)(ctx, idx, center_xyz, center_xyz_batch_cnt, xyz, xyz_batch_cnt, max_radius, sample_num);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBboxOverlapsMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t ious,
                                             diopiConstTensorHandle_t bboxes1,
                                             diopiConstTensorHandle_t bboxes2,
                                             int64_t mode,
                                             int64_t offset,
                                             bool aligned) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBboxOverlapsMmcv"));
    if (func != NULL)
        return (*func)(ctx, ious, bboxes1, bboxes2, mode, offset, aligned);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBorderAlignMmcv(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t output,
                                            diopiTensorHandle_t argmax_idx,
                                            diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t boxes,
                                            int64_t pool_size) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBorderAlignMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, argmax_idx, input, boxes, pool_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBorderAlignBackwardMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t grad_input,
                                                    diopiConstTensorHandle_t grad_output,
                                                    diopiConstTensorHandle_t boxes,
                                                    diopiConstTensorHandle_t argmax_idx,
                                                    int64_t pool_size) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBorderAlignBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, boxes, argmax_idx, pool_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBoxIouRotatedMmcv(diopiContextHandle_t ctx,
                                              diopiTensorHandle_t ious,
                                              diopiConstTensorHandle_t bboxes1,
                                              diopiConstTensorHandle_t bboxes2,
                                              int64_t mode,
                                              bool aligned) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBoxIouRotatedMmcv"));
    if (func != NULL)
        return (*func)(ctx, ious, bboxes1, bboxes2, mode, aligned);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiBoxIouQuadriMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t ious,
                                             diopiConstTensorHandle_t bboxes1,
                                             diopiConstTensorHandle_t bboxes2,
                                             int64_t mode,
                                             bool aligned) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBoxIouQuadriMmcv"));
    if (func != NULL)
        return (*func)(ctx, ious, bboxes1, bboxes2, mode, aligned);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCarafeMmcv(diopiContextHandle_t ctx,
                                       diopiTensorHandle_t rfeatures,
                                       diopiTensorHandle_t routput,
                                       diopiTensorHandle_t rmasks,
                                       diopiTensorHandle_t output,
                                       diopiConstTensorHandle_t features,
                                       diopiConstTensorHandle_t masks,
                                       int64_t kernel_size,
                                       int64_t group_size,
                                       int64_t scale_factor) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCarafeMmcv"));
    if (func != NULL)
        return (*func)(ctx, rfeatures, routput, rmasks, output, features, masks, kernel_size, group_size, scale_factor);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCarafeBackwardMmcv(diopiContextHandle_t ctx,
                                               diopiTensorHandle_t rtop_grad,
                                               diopiTensorHandle_t rbottom_grad_hs,
                                               diopiTensorHandle_t rbottom_grad,
                                               diopiTensorHandle_t rmask_grad,
                                               diopiTensorHandle_t bottom_grad,
                                               diopiTensorHandle_t mask_grad,
                                               diopiConstTensorHandle_t top_grad,
                                               diopiConstTensorHandle_t rfeatures,
                                               diopiConstTensorHandle_t masks,
                                               int64_t kernel_size,
                                               int64_t group_size,
                                               int64_t scale_factor) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCarafeBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, rtop_grad, rbottom_grad_hs, rbottom_grad, rmask_grad, bottom_grad, mask_grad, top_grad, rfeatures, masks, kernel_size, group_size, scale_factor);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCarafeNaiveMmcv(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t output,
                                            diopiConstTensorHandle_t features,
                                            diopiConstTensorHandle_t masks,
                                            int64_t kernel_size,
                                            int64_t group_size,
                                            int64_t scale_factor) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCarafeNaiveMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, features, masks, kernel_size, group_size, scale_factor);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCarafeNaiveBackwardMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t bottom_grad,
                                                    diopiTensorHandle_t mask_grad,
                                                    diopiConstTensorHandle_t top_grad,
                                                    diopiConstTensorHandle_t features,
                                                    diopiConstTensorHandle_t masks,
                                                    int64_t kernel_size,
                                                    int64_t group_size,
                                                    int64_t scale_factor) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCarafeNaiveBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, bottom_grad, mask_grad, top_grad, features, masks, kernel_size, group_size, scale_factor);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCorrelationMmcv(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t output,
                                            diopiConstTensorHandle_t input1,
                                            diopiConstTensorHandle_t input2,
                                            int64_t kH,
                                            int64_t kW,
                                            int64_t patchH,
                                            int64_t patchW,
                                            int64_t padH,
                                            int64_t padW,
                                            int64_t dilationH,
                                            int64_t dilationW,
                                            int64_t dilation_patchH,
                                            int64_t dilation_patchW,
                                            int64_t dH,
                                            int64_t dW) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCorrelationMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input1, input2, kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH, dilation_patchW, dH, dW);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiCorrelationBackwardMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t grad_input1,
                                                    diopiTensorHandle_t grad_input2,
                                                    diopiConstTensorHandle_t grad_output,
                                                    diopiConstTensorHandle_t input1,
                                                    diopiConstTensorHandle_t input2,
                                                    int64_t kH,
                                                    int64_t kW,
                                                    int64_t patchH,
                                                    int64_t patchW,
                                                    int64_t padH,
                                                    int64_t padW,
                                                    int64_t dilationH,
                                                    int64_t dilationW,
                                                    int64_t dilation_patchH,
                                                    int64_t dilation_patchW,
                                                    int64_t dH,
                                                    int64_t dW) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCorrelationBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input1, grad_input2, grad_output, input1, input2, kH, kW, patchH, patchW, padH, padW, dilationH, dilationW, dilation_patchH, dilation_patchW, dH, dW);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDeformConvMmcv(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t output,
                                           diopiTensorHandle_t columns,
                                           diopiTensorHandle_t ones,
                                           diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t offset,
                                           int64_t kW,
                                           int64_t kH,
                                           int64_t dW,
                                           int64_t dH,
                                           int64_t padW,
                                           int64_t padH,
                                           int64_t dilationW,
                                           int64_t dilationH,
                                           int64_t groups,
                                           int64_t deform_groups,
                                           int64_t im2col_step) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDeformConvMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, columns, ones, input, weight, offset, kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deform_groups, im2col_step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDeformConvBackwardInputMmcv(diopiContextHandle_t ctx,
                                                        diopiTensorHandle_t gradInput,
                                                        diopiTensorHandle_t gradOffset,
                                                        diopiConstTensorHandle_t input,
                                                        diopiConstTensorHandle_t offset,
                                                        diopiConstTensorHandle_t gradOutput,
                                                        diopiConstTensorHandle_t weight,
                                                        diopiConstTensorHandle_t columns,
                                                        int64_t kW,
                                                        int64_t kH,
                                                        int64_t dW,
                                                        int64_t dH,
                                                        int64_t padW,
                                                        int64_t padH,
                                                        int64_t dilationW,
                                                        int64_t dilationH,
                                                        int64_t groups,
                                                        int64_t deform_groups,
                                                        int64_t im2col_step) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDeformConvBackwardInputMmcv"));
    if (func != NULL)
        return (*func)(ctx, gradInput, gradOffset, input, offset, gradOutput, weight, columns, kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deform_groups, im2col_step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDeformConvBackwardParametersMmcv(diopiContextHandle_t ctx,
                                                             diopiTensorHandle_t gradOutput,
                                                             diopiTensorHandle_t gradWeight,
                                                             diopiConstTensorHandle_t input,
                                                             diopiConstTensorHandle_t offset,
                                                             diopiConstTensorHandle_t columns,
                                                             diopiConstTensorHandle_t ones,
                                                             int64_t kW,
                                                             int64_t kH,
                                                             int64_t dW,
                                                             int64_t dH,
                                                             int64_t padW,
                                                             int64_t padH,
                                                             int64_t dilationW,
                                                             int64_t dilationH,
                                                             int64_t groups,
                                                             int64_t deform_groups,
                                                             int64_t im2col_step,
                                                             float scale) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDeformConvBackwardParametersMmcv"));
    if (func != NULL)
        return (*func)(ctx, gradOutput, gradWeight, input, offset, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups, deform_groups, im2col_step, scale);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDeformRoiPoolMmcv(diopiContextHandle_t ctx,
                                              diopiTensorHandle_t output,
                                              diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t rois,
                                              diopiConstTensorHandle_t offset,
                                              int64_t pooled_height,
                                              int64_t pooled_width,
                                              int64_t sampling_ratio,
                                              float spatial_scale,
                                              float gamma) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDeformRoiPoolMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input, rois, offset, pooled_height, pooled_width, sampling_ratio, spatial_scale, gamma);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDeformRoiPoolBackwardMmcv(diopiContextHandle_t ctx,
                                                      diopiTensorHandle_t grad_input,
                                                      diopiTensorHandle_t grad_offset,
                                                      diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input,
                                                      diopiConstTensorHandle_t rois,
                                                      diopiConstTensorHandle_t offset,
                                                      int64_t pooled_height,
                                                      int64_t pooled_width,
                                                      int64_t sampling_ratio,
                                                      float spatial_scale,
                                                      float gamma) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDeformRoiPoolBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_offset, grad_output, input, rois, offset, pooled_height, pooled_width, sampling_ratio, spatial_scale, gamma);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t output,
                                                 diopiConstTensorHandle_t input,
                                                 diopiConstTensorHandle_t target,
                                                 diopiConstTensorHandle_t weight,
                                                 float gamma,
                                                 float alpha) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidFocalLossMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input, target, weight, gamma, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx,
                                                         diopiTensorHandle_t grad_input,
                                                         diopiConstTensorHandle_t input,
                                                         diopiConstTensorHandle_t target,
                                                         diopiConstTensorHandle_t weight,
                                                         float gamma,
                                                         float alpha) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidFocalLossBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, input, target, weight, gamma, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSoftmaxFocalLossMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t output,
                                                 diopiConstTensorHandle_t softmax,
                                                 diopiConstTensorHandle_t target,
                                                 diopiConstTensorHandle_t weight,
                                                 float gamma,
                                                 float alpha) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSoftmaxFocalLossMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, softmax, target, weight, gamma, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSoftmaxFocalLossBackwardMmcv(diopiContextHandle_t ctx,
                                                         diopiTensorHandle_t grad_input,
                                                         diopiTensorHandle_t buff,
                                                         diopiConstTensorHandle_t softmax,
                                                         diopiConstTensorHandle_t target,
                                                         diopiConstTensorHandle_t weight,
                                                         float gamma,
                                                         float alpha) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSoftmaxFocalLossBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, buff, softmax, target, weight, gamma, alpha);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFurthestPointSamplingMmcv(diopiContextHandle_t ctx,
                                                      diopiTensorHandle_t temp_tensor,
                                                      diopiTensorHandle_t idx_tensor,
                                                      diopiConstTensorHandle_t points_xyz,
                                                      int64_t B,
                                                      int64_t N,
                                                      int64_t num_points) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFurthestPointSamplingMmcv"));
    if (func != NULL)
        return (*func)(ctx, temp_tensor, idx_tensor, points_xyz, B, N, num_points);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFurthestPointSamplingWithDistMmcv(diopiContextHandle_t ctx,
                                                              diopiTensorHandle_t temp_tensor,
                                                              diopiTensorHandle_t idx_tensor,
                                                              diopiConstTensorHandle_t points_dist,
                                                              int64_t B,
                                                              int64_t N,
                                                              int64_t num_points) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFurthestPointSamplingWithDistMmcv"));
    if (func != NULL)
        return (*func)(ctx, temp_tensor, idx_tensor, points_dist, B, N, num_points);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFusedBiasLeakyreluMmcv(diopiContextHandle_t ctx,
                                                   diopiTensorHandle_t* out,
                                                   diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t bias,
                                                   diopiConstTensorHandle_t refer,
                                                   int64_t act,
                                                   int64_t grad,
                                                   float alpha,
                                                   float scale) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        float,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFusedBiasLeakyreluMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, input, bias, refer, act, grad, alpha, scale);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGatherPointsMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t out,
                                             diopiConstTensorHandle_t points,
                                             diopiConstTensorHandle_t idx,
                                             int64_t b,
                                             int64_t c,
                                             int64_t n,
                                             int64_t npoints) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGatherPointsMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, points, idx, b, c, n, npoints);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGatherPointsBackwardMmcv(diopiContextHandle_t ctx,
                                                     diopiTensorHandle_t grad_points,
                                                     diopiConstTensorHandle_t grad_out,
                                                     diopiConstTensorHandle_t idx,
                                                     int64_t b,
                                                     int64_t c,
                                                     int64_t n,
                                                     int64_t npoints) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGatherPointsBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_points, grad_out, idx, b, c, n, npoints);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGroupPointsMmcv(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t out,
                                            diopiConstTensorHandle_t points,
                                            diopiConstTensorHandle_t idx,
                                            int64_t b,
                                            int64_t c,
                                            int64_t n,
                                            int64_t npoints,
                                            int64_t nsample) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGroupPointsMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, points, idx, b, c, n, npoints, nsample);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiGroupPointsBackwardMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t grad_points,
                                                    diopiConstTensorHandle_t grad_out,
                                                    diopiConstTensorHandle_t idx,
                                                    int64_t b,
                                                    int64_t c,
                                                    int64_t n,
                                                    int64_t npoints,
                                                    int64_t nsample) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGroupPointsBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_points, grad_out, idx, b, c, n, npoints, nsample);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiStackGroupPointsMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t out_tensor,
                                                 diopiConstTensorHandle_t features_tensor,
                                                 diopiConstTensorHandle_t features_batch_cnt_tensor,
                                                 diopiConstTensorHandle_t idx_tensor,
                                                 diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                                 int64_t b,
                                                 int64_t c,
                                                 int64_t m,
                                                 int64_t nsample) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiStackGroupPointsMmcv"));
    if (func != NULL)
        return (*func)(ctx, out_tensor, features_tensor, features_batch_cnt_tensor, idx_tensor, idx_batch_cnt_tensor, b, c, m, nsample);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiStackGroupPointsBackwardMmcv(diopiContextHandle_t ctx,
                                                         diopiTensorHandle_t grad_features_tensor,
                                                         diopiConstTensorHandle_t grad_out_tensor,
                                                         diopiConstTensorHandle_t idx_tensor,
                                                         diopiConstTensorHandle_t idx_batch_cnt_tensor,
                                                         diopiConstTensorHandle_t features_batch_cnt_tensor,
                                                         int64_t b,
                                                         int64_t c,
                                                         int64_t m,
                                                         int64_t n,
                                                         int64_t nsample) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiStackGroupPointsBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_features_tensor, grad_out_tensor, idx_tensor, idx_batch_cnt_tensor, features_batch_cnt_tensor, b, c, m, n, nsample);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIou3dBoxesOverlapBevMmcv(diopiContextHandle_t ctx,
                                                     diopiTensorHandle_t ans_overlap,
                                                     diopiConstTensorHandle_t boxes_a,
                                                     diopiConstTensorHandle_t boxes_b) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIou3dBoxesOverlapBevMmcv"));
    if (func != NULL)
        return (*func)(ctx, ans_overlap, boxes_a, boxes_b);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIou3dNms3dMmcv(
    diopiContextHandle_t ctx, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, diopiConstTensorHandle_t boxes, float nms_overlap_thresh) {
    diopiError_t (*func)(
        diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIou3dNms3dMmcv"));
    if (func != NULL)
        return (*func)(ctx, keep, keep_num, boxes, nms_overlap_thresh);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIou3dNms3dNormalMmcv(
    diopiContextHandle_t ctx, diopiTensorHandle_t keep, diopiTensorHandle_t keep_num, diopiConstTensorHandle_t boxes, float nms_overlap_thresh) {
    diopiError_t (*func)(
        diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIou3dNms3dNormalMmcv"));
    if (func != NULL)
        return (*func)(ctx, keep, keep_num, boxes, nms_overlap_thresh);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiKnnMmcv(diopiContextHandle_t ctx,
                                    diopiTensorHandle_t idx_tensor,
                                    diopiTensorHandle_t dist2_tensor,
                                    diopiConstTensorHandle_t xyz_tensor,
                                    diopiConstTensorHandle_t new_xyz_tensor,
                                    int64_t b,
                                    int64_t n,
                                    int64_t m,
                                    int64_t nsample) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiKnnMmcv"));
    if (func != NULL)
        return (*func)(ctx, idx_tensor, dist2_tensor, xyz_tensor, new_xyz_tensor, b, n, m, nsample);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedIm2colMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t col,
                                             diopiConstTensorHandle_t im,
                                             diopiConstTensorHandle_t mask_h_idx,
                                             diopiConstTensorHandle_t mask_w_idx,
                                             int64_t kernel_h,
                                             int64_t kernel_w,
                                             int64_t pad_h,
                                             int64_t pad_w) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedIm2colMmcv"));
    if (func != NULL)
        return (*func)(ctx, col, im, mask_h_idx, mask_w_idx, kernel_h, kernel_w, pad_h, pad_w);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMaskedCol2imMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t im,
                                             diopiConstTensorHandle_t col,
                                             diopiConstTensorHandle_t mask_h_idx,
                                             diopiConstTensorHandle_t mask_w_idx,
                                             int64_t height,
                                             int64_t width,
                                             int64_t channels) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedCol2imMmcv"));
    if (func != NULL)
        return (*func)(ctx, im, col, mask_h_idx, mask_w_idx, height, width, channels);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiModulatedDeformConvMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t output,
                                                    diopiTensorHandle_t columns,
                                                    diopiTensorHandle_t ones,
                                                    diopiConstTensorHandle_t input,
                                                    diopiConstTensorHandle_t weight,
                                                    diopiConstTensorHandle_t bias,
                                                    diopiConstTensorHandle_t offset,
                                                    diopiConstTensorHandle_t mask,
                                                    int64_t kernel_h,
                                                    int64_t kernel_w,
                                                    const int64_t stride_h,
                                                    const int64_t stride_w,
                                                    const int64_t pad_h,
                                                    const int64_t pad_w,
                                                    const int64_t dilation_h,
                                                    const int64_t dilation_w,
                                                    const int64_t group,
                                                    const int64_t deformable_group,
                                                    const bool with_bias) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        const int64_t,
        const int64_t,
        const int64_t,
        const int64_t,
        const int64_t,
        const int64_t,
        const int64_t,
        const int64_t,
        const bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiModulatedDeformConvMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, columns, ones, input, weight, bias, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, deformable_group, with_bias);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiModulatedDeformConvBackwardMmcv(diopiContextHandle_t ctx,
                                                            diopiTensorHandle_t grad_input,
                                                            diopiTensorHandle_t grad_weight,
                                                            diopiTensorHandle_t grad_bias,
                                                            diopiTensorHandle_t grad_offset,
                                                            diopiTensorHandle_t grad_mask,
                                                            diopiConstTensorHandle_t input,
                                                            diopiConstTensorHandle_t weight,
                                                            diopiConstTensorHandle_t bias,
                                                            diopiConstTensorHandle_t ones,
                                                            diopiConstTensorHandle_t offset,
                                                            diopiConstTensorHandle_t mask,
                                                            diopiConstTensorHandle_t columns,
                                                            diopiConstTensorHandle_t grad_output,
                                                            int64_t kernel_h,
                                                            int64_t kernel_w,
                                                            int64_t stride_h,
                                                            int64_t stride_w,
                                                            int64_t pad_h,
                                                            int64_t pad_w,
                                                            int64_t dilation_h,
                                                            int64_t dilation_w,
                                                            int64_t group,
                                                            int64_t deformable_group,
                                                            const bool with_bias) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        const bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiModulatedDeformConvBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_offset, grad_mask, input, weight, bias, ones, offset, mask, columns, grad_output, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, deformable_group, with_bias);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMsDeformAttnMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t* out,
                                             diopiConstTensorHandle_t value,
                                             diopiConstTensorHandle_t spatial_shapes,
                                             diopiConstTensorHandle_t level_start_index,
                                             diopiConstTensorHandle_t sampling_loc,
                                             diopiConstTensorHandle_t attn_weight,
                                             int64_t im2col_step) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMsDeformAttnMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMsDeformAttnBackwardMmcv(diopiContextHandle_t ctx,
                                                     diopiTensorHandle_t grad_value,
                                                     diopiTensorHandle_t grad_sampling_loc,
                                                     diopiTensorHandle_t grad_attn_weight,
                                                     diopiConstTensorHandle_t value,
                                                     diopiConstTensorHandle_t spatial_shapes,
                                                     diopiConstTensorHandle_t level_start_index,
                                                     diopiConstTensorHandle_t sampling_loc,
                                                     diopiConstTensorHandle_t attn_weight,
                                                     diopiConstTensorHandle_t grad_output,
                                                     int64_t im2col_step) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMsDeformAttnBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_value, grad_sampling_loc, grad_attn_weight, value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx,
                                    diopiTensorHandle_t* out,
                                    diopiConstTensorHandle_t dets,
                                    diopiConstTensorHandle_t scores,
                                    double iou_threshold,
                                    int64_t offset) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        double,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNmsMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, dets, scores, iou_threshold, offset);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiNmsRotatedMmcv(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t* out,
                                           diopiConstTensorHandle_t dets,
                                           diopiConstTensorHandle_t scores,
                                           diopiConstTensorHandle_t order_t,
                                           diopiConstTensorHandle_t dets_sorted,
                                           double iou_threshold,
                                           int64_t multi_label) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        double,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNmsRotatedMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, dets, scores, order_t, dets_sorted, iou_threshold, multi_label);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPointsInBoxesPartMmcv(diopiContextHandle_t ctx,
                                                  diopiTensorHandle_t box_idx_of_points,
                                                  diopiConstTensorHandle_t boxes,
                                                  diopiConstTensorHandle_t pts) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPointsInBoxesPartMmcv"));
    if (func != NULL)
        return (*func)(ctx, box_idx_of_points, boxes, pts);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPointsInBoxesAllMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t box_idx_of_points,
                                                 diopiConstTensorHandle_t boxes,
                                                 diopiConstTensorHandle_t pts) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPointsInBoxesAllMmcv"));
    if (func != NULL)
        return (*func)(ctx, box_idx_of_points, boxes, pts);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPsamaskMmcv(diopiContextHandle_t ctx,
                                        diopiTensorHandle_t output,
                                        diopiConstTensorHandle_t input,
                                        int64_t psa_type,
                                        int64_t num_,
                                        int64_t h_feature,
                                        int64_t w_feature,
                                        int64_t h_mask,
                                        int64_t w_mask,
                                        int64_t half_h_mask,
                                        int64_t half_w_mask) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPsamaskMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input, psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask, half_w_mask);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPsamaskBackwardMmcv(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t grad_input,
                                                diopiConstTensorHandle_t grad_output,
                                                int64_t psa_type,
                                                int64_t num_,
                                                int64_t h_feature,
                                                int64_t w_feature,
                                                int64_t h_mask,
                                                int64_t w_mask,
                                                int64_t half_h_mask,
                                                int64_t half_w_mask) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPsamaskBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask, half_w_mask);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx,
                                         diopiTensorHandle_t output,
                                         diopiTensorHandle_t argmax_y,
                                         diopiTensorHandle_t argmax_x,
                                         diopiConstTensorHandle_t input,
                                         diopiConstTensorHandle_t rois,
                                         int64_t aligned_height,
                                         int64_t aligned_width,
                                         int64_t sampling_ratio,
                                         int64_t pool_mode,
                                         float spatial_scale,
                                         bool aligned) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlignMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, argmax_y, argmax_x, input, rois, aligned_height, aligned_width, sampling_ratio, pool_mode, spatial_scale, aligned);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t grad_input,
                                                 diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t rois,
                                                 diopiConstTensorHandle_t argmax_y,
                                                 diopiConstTensorHandle_t argmax_x,
                                                 int64_t aligned_height,
                                                 int64_t aligned_width,
                                                 int64_t sampling_ratio,
                                                 int64_t pool_mode,
                                                 float spatial_scale,
                                                 bool aligned) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlignBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, rois, argmax_y, argmax_x, aligned_height, aligned_width, sampling_ratio, pool_mode, spatial_scale, aligned);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiAlignRotatedMmcv(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t output,
                                                diopiConstTensorHandle_t input,
                                                diopiConstTensorHandle_t rois,
                                                int64_t aligned_height,
                                                int64_t aligned_width,
                                                int64_t sampling_ratio,
                                                float spatial_scale,
                                                bool aligned,
                                                bool clockwise) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        bool,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlignRotatedMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input, rois, aligned_height, aligned_width, sampling_ratio, spatial_scale, aligned, clockwise);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiAlignRotatedBackwardMmcv(diopiContextHandle_t ctx,
                                                        diopiTensorHandle_t bottom_grad,
                                                        diopiConstTensorHandle_t top_grad,
                                                        diopiConstTensorHandle_t rois,
                                                        int64_t aligned_height,
                                                        int64_t aligned_width,
                                                        int64_t sampling_ratio,
                                                        float spatial_scale,
                                                        bool aligned,
                                                        bool clockwise) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        bool,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlignRotatedBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, bottom_grad, top_grad, rois, aligned_height, aligned_width, sampling_ratio, spatial_scale, aligned, clockwise);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRiroiAlignRotatedMmcv(diopiContextHandle_t ctx,
                                                  diopiTensorHandle_t output,
                                                  diopiConstTensorHandle_t features,
                                                  diopiConstTensorHandle_t rois,
                                                  int64_t pooled_height,
                                                  int64_t pooled_width,
                                                  int64_t num_samples,
                                                  int64_t num_orientations,
                                                  float spatial_scale,
                                                  bool clockwise) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRiroiAlignRotatedMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, features, rois, pooled_height, pooled_width, num_samples, num_orientations, spatial_scale, clockwise);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRiroiAlignRotatedBackwardMmcv(diopiContextHandle_t ctx,
                                                          diopiTensorHandle_t bottom_grad,
                                                          diopiConstTensorHandle_t top_grad,
                                                          diopiConstTensorHandle_t rois,
                                                          int64_t pooled_height,
                                                          int64_t pooled_width,
                                                          int64_t num_samples,
                                                          int64_t num_orientations,
                                                          float spatial_scale,
                                                          bool clockwise) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        float,
        bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRiroiAlignRotatedBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, bottom_grad, top_grad, rois, pooled_height, pooled_width, num_samples, num_orientations, spatial_scale, clockwise);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiawarePool3dMmcv(diopiContextHandle_t ctx,
                                               diopiTensorHandle_t argmax,
                                               diopiTensorHandle_t pts_idx_of_voxels,
                                               diopiTensorHandle_t pooled_features,
                                               diopiConstTensorHandle_t rois,
                                               diopiConstTensorHandle_t pts,
                                               diopiConstTensorHandle_t pts_feature,
                                               int64_t pool_method) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiawarePool3dMmcv"));
    if (func != NULL)
        return (*func)(ctx, argmax, pts_idx_of_voxels, pooled_features, rois, pts, pts_feature, pool_method);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiawarePool3dBackwardMmcv(diopiContextHandle_t ctx,
                                                       diopiTensorHandle_t grad_in,
                                                       diopiConstTensorHandle_t pts_idx_of_voxels,
                                                       diopiConstTensorHandle_t argmax,
                                                       diopiConstTensorHandle_t grad_out,
                                                       int64_t pool_method) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiawarePool3dBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_in, pts_idx_of_voxels, argmax, grad_out, pool_method);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoipointPool3dMmcv(diopiContextHandle_t ctx,
                                               diopiTensorHandle_t pooled_features,
                                               diopiTensorHandle_t pooled_empty_flag,
                                               diopiConstTensorHandle_t xyz,
                                               diopiConstTensorHandle_t boxes3d,
                                               diopiConstTensorHandle_t pts_feature) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoipointPool3dMmcv"));
    if (func != NULL)
        return (*func)(ctx, pooled_features, pooled_empty_flag, xyz, boxes3d, pts_feature);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiPoolMmcv(diopiContextHandle_t ctx,
                                        diopiTensorHandle_t output,
                                        diopiTensorHandle_t argmax,
                                        diopiConstTensorHandle_t input,
                                        diopiConstTensorHandle_t rois,
                                        int64_t pooled_height,
                                        int64_t pooled_width,
                                        float spatial_scale) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiPoolMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, argmax, input, rois, pooled_height, pooled_width, spatial_scale);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRoiPoolBackwardMmcv(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t grad_input,
                                                diopiConstTensorHandle_t grad_output,
                                                diopiConstTensorHandle_t rois,
                                                diopiConstTensorHandle_t argmax,
                                                int64_t pooled_height,
                                                int64_t pooled_width,
                                                float spatial_scale) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiPoolBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, rois, argmax, pooled_height, pooled_width, spatial_scale);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDynamicPointToVoxelMmcv(
    diopiContextHandle_t ctx, diopiTensorHandle_t* outlist, diopiConstTensorHandle_t feats, diopiConstTensorHandle_t coors, int64_t reduce_type) {
    diopiError_t (*func)(
        diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDynamicPointToVoxelMmcv"));
    if (func != NULL)
        return (*func)(ctx, outlist, feats, coors, reduce_type);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDynamicPointToVoxelBackwardMmcv(diopiContextHandle_t ctx,
                                                            diopiTensorHandle_t grad_feats,
                                                            diopiConstTensorHandle_t grad_reduced_feats,
                                                            diopiConstTensorHandle_t feats,
                                                            diopiConstTensorHandle_t reduced_feats,
                                                            diopiConstTensorHandle_t coors_idx,
                                                            diopiConstTensorHandle_t reduce_count,
                                                            int64_t reduce_type) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDynamicPointToVoxelBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_feats, grad_reduced_feats, feats, reduced_feats, coors_idx, reduce_count, reduce_type);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSyncBnMeanMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiConstTensorHandle_t input) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSyncBnMeanMmcv"));
    if (func != NULL)
        return (*func)(ctx, mean, input);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSyncBnVarMmcv(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t var,
                                          diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t mean) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSyncBnVarMmcv"));
    if (func != NULL)
        return (*func)(ctx, var, input, mean);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSyncBnOutputMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t running_mean,
                                             diopiTensorHandle_t running_var,
                                             diopiTensorHandle_t norm,
                                             diopiTensorHandle_t std,
                                             diopiTensorHandle_t output,
                                             diopiConstTensorHandle_t input,
                                             diopiConstTensorHandle_t mean,
                                             diopiConstTensorHandle_t var,
                                             diopiConstTensorHandle_t weight,
                                             diopiConstTensorHandle_t bias,
                                             float eps,
                                             float momentum,
                                             int64_t group_size) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        float,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSyncBnOutputMmcv"));
    if (func != NULL)
        return (*func)(ctx, running_mean, running_var, norm, std, output, input, mean, var, weight, bias, eps, momentum, group_size);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSyncBnBackwardParamMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t grad_weight,
                                                    diopiTensorHandle_t grad_bias,
                                                    diopiConstTensorHandle_t grad_output,
                                                    diopiConstTensorHandle_t norm) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSyncBnBackwardParamMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_weight, grad_bias, grad_output, norm);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSyncBnBackwardDataMmcv(diopiContextHandle_t ctx,
                                                   diopiTensorHandle_t grad_input,
                                                   diopiConstTensorHandle_t grad_output,
                                                   diopiConstTensorHandle_t weight,
                                                   diopiConstTensorHandle_t grad_weight,
                                                   diopiConstTensorHandle_t grad_bias,
                                                   diopiConstTensorHandle_t norm,
                                                   diopiConstTensorHandle_t std) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSyncBnBackwardDataMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, weight, grad_weight, grad_bias, norm, std);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiThreeInterpolateMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t out,
                                                 diopiConstTensorHandle_t points,
                                                 diopiConstTensorHandle_t idx,
                                                 diopiConstTensorHandle_t weight,
                                                 int64_t b,
                                                 int64_t c,
                                                 int64_t m,
                                                 int64_t n) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThreeInterpolateMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, points, idx, weight, b, c, m, n);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiThreeInterpolateBackwardMmcv(diopiContextHandle_t ctx,
                                                         diopiTensorHandle_t grad_points,
                                                         diopiConstTensorHandle_t grad_out,
                                                         diopiConstTensorHandle_t idx,
                                                         diopiConstTensorHandle_t weight,
                                                         int64_t b,
                                                         int64_t c,
                                                         int64_t n,
                                                         int64_t m) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThreeInterpolateBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_points, grad_out, idx, weight, b, c, n, m);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiThreeNnMmcv(diopiContextHandle_t ctx,
                                        diopiTensorHandle_t dist2,
                                        diopiTensorHandle_t idx,
                                        diopiConstTensorHandle_t unknown,
                                        diopiConstTensorHandle_t known,
                                        int64_t b,
                                        int64_t n,
                                        int64_t m) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThreeNnMmcv"));
    if (func != NULL)
        return (*func)(ctx, dist2, idx, unknown, known, b, n, m);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTinShiftMmcv(diopiContextHandle_t ctx,
                                         diopiTensorHandle_t output,
                                         diopiConstTensorHandle_t input,
                                         diopiConstTensorHandle_t shift) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTinShiftMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input, shift);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiTinShiftBackwardMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t grad_input,
                                                 diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t shift) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTinShiftBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, shift);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiUpfirdn2dOpMmcv(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t* out,
                                            diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t kernel,
                                            int64_t up_x,
                                            int64_t up_y,
                                            int64_t down_x,
                                            int64_t down_y,
                                            int64_t pad_x0,
                                            int64_t pad_x1,
                                            int64_t pad_y0,
                                            int64_t pad_y1) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpfirdn2dOpMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiHardVoxelizeMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t voxels,
                                             diopiTensorHandle_t coors,
                                             diopiTensorHandle_t num_points_per_voxel,
                                             diopiTensorHandle_t voxel_num,
                                             diopiConstTensorHandle_t points,
                                             diopiConstTensorHandle_t voxel_size,
                                             diopiConstTensorHandle_t coors_range,
                                             const int64_t max_points,
                                             const int64_t max_voxels,
                                             const int64_t NDim,
                                             const bool deterministic) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        const int64_t,
        const int64_t,
        const int64_t,
        const bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardVoxelizeMmcv"));
    if (func != NULL)
        return (*func)(ctx, voxels, coors, num_points_per_voxel, voxel_num, points, voxel_size, coors_range, max_points, max_voxels, NDim, deterministic);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDynamicVoxelizeMmcv(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t coors,
                                                diopiConstTensorHandle_t points,
                                                diopiConstTensorHandle_t voxel_size,
                                                diopiConstTensorHandle_t coors_range,
                                                const int64_t NDim) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        const int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDynamicVoxelizeMmcv"));
    if (func != NULL)
        return (*func)(ctx, coors, points, voxel_size, coors_range, NDim);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRotatedFeatureAlignMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t output,
                                                    diopiConstTensorHandle_t features,
                                                    diopiConstTensorHandle_t best_bboxes,
                                                    float spatial_scale,
                                                    int64_t points) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRotatedFeatureAlignMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, features, best_bboxes, spatial_scale, points);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiRotatedFeatureAlignBackwardMmcv(diopiContextHandle_t ctx,
                                                            diopiTensorHandle_t bottom_grad,
                                                            diopiConstTensorHandle_t top_grad,
                                                            diopiConstTensorHandle_t best_bboxes,
                                                            float spatial_scale,
                                                            int64_t points) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        float,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRotatedFeatureAlignBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, bottom_grad, top_grad, best_bboxes, spatial_scale, points);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPointsInPolygonsMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t output,
                                                 diopiConstTensorHandle_t points,
                                                 diopiConstTensorHandle_t polygons) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPointsInPolygonsMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, points, polygons);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndiceMaxpoolMmcv(diopiContextHandle_t ctx,
                                              diopiTensorHandle_t* out,
                                              diopiConstTensorHandle_t features,
                                              diopiConstTensorHandle_t indicePairs,
                                              diopiConstTensorHandle_t indiceNum,
                                              int64_t numAct) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndiceMaxpoolMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, features, indicePairs, indiceNum, numAct);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndiceMaxpoolBackwardMmcv(diopiContextHandle_t ctx,
                                                      diopiTensorHandle_t* out,
                                                      diopiConstTensorHandle_t features,
                                                      diopiConstTensorHandle_t outFeatures,
                                                      diopiConstTensorHandle_t outGrad,
                                                      diopiConstTensorHandle_t indicePairs,
                                                      diopiConstTensorHandle_t indiceNum) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndiceMaxpoolBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, features, outFeatures, outGrad, indicePairs, indiceNum);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndiceConvMmcv(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t* out,
                                           diopiConstTensorHandle_t features,
                                           diopiConstTensorHandle_t filters,
                                           diopiConstTensorHandle_t indicePairs,
                                           diopiConstTensorHandle_t indiceNum,
                                           int64_t numActOut,
                                           int64_t _inverse,
                                           int64_t _subM) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndiceConvMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiIndiceConvBackwardMmcv(diopiContextHandle_t ctx,
                                                   diopiTensorHandle_t* outlist,
                                                   diopiConstTensorHandle_t features,
                                                   diopiConstTensorHandle_t filters,
                                                   diopiConstTensorHandle_t outGrad,
                                                   diopiConstTensorHandle_t indicePairs,
                                                   diopiConstTensorHandle_t indiceNum,
                                                   int64_t _inverse,
                                                   int64_t _subM) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndiceConvBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, outlist, features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFusedIndiceConvBatchnormMmcv(diopiContextHandle_t ctx,
                                                         diopiTensorHandle_t* out,
                                                         diopiConstTensorHandle_t features,
                                                         diopiConstTensorHandle_t filters,
                                                         diopiConstTensorHandle_t bias,
                                                         diopiConstTensorHandle_t indicePairs,
                                                         diopiConstTensorHandle_t indiceNum,
                                                         int64_t numActOut,
                                                         int64_t _inverse,
                                                         int64_t _subM) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFusedIndiceConvBatchnormMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, features, filters, bias, indicePairs, indiceNum, numActOut, _inverse, _subM);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiMinAreaPolygonsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t polygons, diopiConstTensorHandle_t pointsets) {
    diopiError_t (*func)(diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMinAreaPolygonsMmcv"));
    if (func != NULL)
        return (*func)(ctx, polygons, pointsets);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiActiveRotatedFilterMmcv(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t output,
                                                    diopiConstTensorHandle_t input,
                                                    diopiConstTensorHandle_t indices) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiActiveRotatedFilterMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input, indices);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiActiveRotatedFilterBackwardMmcv(diopiContextHandle_t ctx,
                                                            diopiTensorHandle_t grad_in,
                                                            diopiConstTensorHandle_t grad_out,
                                                            diopiConstTensorHandle_t indices) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiActiveRotatedFilterBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_in, grad_out, indices);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiConvexIouMmcv(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t ious,
                                          diopiConstTensorHandle_t pointsets,
                                          diopiConstTensorHandle_t polygons) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvexIouMmcv"));
    if (func != NULL)
        return (*func)(ctx, ious, pointsets, polygons);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiConvexGiouMmcv(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t output,
                                           diopiConstTensorHandle_t pointsets,
                                           diopiConstTensorHandle_t polygons) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvexGiouMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, pointsets, polygons);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiDiffIouRotatedSortVerticesMmcv(diopiContextHandle_t ctx,
                                                           diopiTensorHandle_t* out,
                                                           diopiConstTensorHandle_t vertices,
                                                           diopiConstTensorHandle_t mask,
                                                           diopiConstTensorHandle_t num_valid) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t*,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDiffIouRotatedSortVerticesMmcv"));
    if (func != NULL)
        return (*func)(ctx, out, vertices, mask, num_valid);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiChamferDistanceMmcv(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t dist1,
                                                diopiTensorHandle_t dist2,
                                                diopiTensorHandle_t idx1,
                                                diopiTensorHandle_t idx2,
                                                diopiConstTensorHandle_t xyz1,
                                                diopiConstTensorHandle_t xyz2) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiChamferDistanceMmcv"));
    if (func != NULL)
        return (*func)(ctx, dist1, dist2, idx1, idx2, xyz1, xyz2);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiChamferDistanceBackwardMmcv(diopiContextHandle_t ctx,
                                                        diopiTensorHandle_t grad_xyz1,
                                                        diopiTensorHandle_t grad_xyz2,
                                                        diopiConstTensorHandle_t xyz1,
                                                        diopiConstTensorHandle_t xyz2,
                                                        diopiConstTensorHandle_t idx1,
                                                        diopiConstTensorHandle_t idx2,
                                                        diopiConstTensorHandle_t grad_dist1,
                                                        diopiConstTensorHandle_t grad_dist2) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiChamferDistanceBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_xyz1, grad_xyz2, xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPrroiPoolMmcv(diopiContextHandle_t ctx,
                                          diopiTensorHandle_t output,
                                          diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t rois,
                                          int64_t pooled_height,
                                          int64_t pooled_width,
                                          float spatial_scale) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPrroiPoolMmcv"));
    if (func != NULL)
        return (*func)(ctx, output, input, rois, pooled_height, pooled_width, spatial_scale);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPrroiPoolbackwardMmcv(diopiContextHandle_t ctx,
                                                  diopiTensorHandle_t grad_input,
                                                  diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t rois,
                                                  int64_t pooled_height,
                                                  int64_t pooled_width,
                                                  float spatial_scale) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPrroiPoolbackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_input, grad_output, rois, pooled_height, pooled_width, spatial_scale);
    else
        return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiPrroiPoolCoorBackwardMmcv(diopiContextHandle_t ctx,
                                                      diopiTensorHandle_t grad_rois,
                                                      diopiConstTensorHandle_t output,
                                                      diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input,
                                                      diopiConstTensorHandle_t rois,
                                                      int64_t pooled_height,
                                                      int64_t pooled_width,
                                                      float spatial_scale) {
    diopiError_t (*func)(diopiContextHandle_t,
        diopiTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        diopiConstTensorHandle_t,
        int64_t,
        int64_t,
        float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPrroiPoolCoorBackwardMmcv"));
    if (func != NULL)
        return (*func)(ctx, grad_rois, output, grad_output, input, rois, pooled_height, pooled_width, spatial_scale);
    else
        return diopiErrorOccurred;
}


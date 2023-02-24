#include <diopi/functions.h>
#include <cnnl.h>

#include <cstdio>
#include <vector>

#include "helper.hpp"

#define DIOPI_CALLCNNL(Expr) { \
        ::cnnlStatus_t ret = Expr; \
        if (ret != ::CNNL_STATUS_SUCCESS) { \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s",           \
                    ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__);               \
            return diopiErrorOccurred;                                                 \
        }}\

#define DIOPI_CHECKCNNL(Expr) { \
        ::cnnlStatus_t ret = Expr; \
        if (ret != ::CNNL_STATUS_SUCCESS) { \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s",           \
                    ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__);               \
        }}\

#define DIOPI_CHECK(cond, str) \
    if (!(cond)) { \
        impl::camb::set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
        return diopiErrorOccurred; \
    } \


template<typename T, ::cnnlStatus_t(*fnCreate)(T*), ::cnnlStatus_t(*fnDestroy)(T)>
class CnnlResourceGuard {
public:
    CnnlResourceGuard() {
        DIOPI_CHECKCNNL(fnCreate(&resource_));
    }

    ~CnnlResourceGuard() {
        DIOPI_CHECKCNNL(fnDestroy(resource_));
    }

    T& get() {
        return resource_;
    }

protected:
    T resource_ {0};
};

class CnnlTransposeDescriptor final
    : public CnnlResourceGuard<cnnlTransposeDescriptor_t,
          cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> {
public:
    CnnlTransposeDescriptor() {}

    CnnlTransposeDescriptor(const int dim, const int* permute) {
        set(dim, permute);
    }

    diopiError_t set(const int dim, const int* permute) {
        DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(get(), dim, permute));
        return diopiSuccess;
    }
};

class CnnlTensorDescriptor final : public CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor>
{
public:
  CnnlTensorDescriptor() {}
  CnnlTensorDescriptor(auto &t, cnnlTensorLayout_t layout) {
    set(t, layout);
  }
  template<typename T>
  diopiError_t set(T& t, cnnlTensorLayout_t layout) {
    int dimNb = t.dim();
    auto dimSize = t.shape().data;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, t.dtype()));

    std::vector<int> shape_info(dimNb);
    if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC
            || layout == CNNL_LAYOUT_NLC) {
        shape_info[0] = dimSize[0];
        for (size_t i = 0; i < dimNb - 1; ++i) {
            shape_info[i+1] = dimSize[(i + 1) % (dimNb - 1) + 1];
        }
    } else if (layout == CNNL_LAYOUT_HWCN) {
        // HWCN is only used by depthwise conv now, and the dim is 4
        DIOPI_CHECK(dimNb == 4, "depthwise convolution input's dim must be 4!");
        shape_info[0] = dimSize[2];
        shape_info[1] = dimSize[3];
        shape_info[2] = dimSize[1];
        shape_info[3] = dimSize[0];
    } else {
        for (size_t i = 0; i < dimNb; ++i) {
            shape_info[i] = dimSize[i];
        }
    }
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(this->get(), layout,
                                                dtype, dimNb, shape_info.data()));
    return diopiSuccess;
}
};

template<typename T>
static diopiError_t cnnl_transpose(diopiContextHandle_t& ctx, cnnlHandle_t& handle, T& in, impl::camb::DiopiTensor<diopiTensorHandle_t>& out, cnnlTensorLayout_t layoutIn,
                          cnnlTensorLayout_t layoutOut) {
    std::vector<int> order;
    if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {1, 2, 3, 0};
    } else if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {0, 3, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {2, 3, 1, 0};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {0, 2, 3, 1};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {3, 0, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {3, 2, 0, 1};
    } else {
        impl::camb::set_last_error_string("unkown layout error, layout should be in [CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_HWCN], at %s:%s", __FILE__, __LINE__);
        return diopiDtypeNotSupported;
    }
    CnnlTensorDescriptor inDesc(in, layoutIn);
    CnnlTensorDescriptor outDesc(out, layoutOut);
    CnnlTransposeDescriptor transDesc(order.size(), order.data());
    size_t workspace_size = 0;
    DIOPI_CHECKCNNL(cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), transDesc.get(), &workspace_size));

    void* workspace_ptr = workspace_size== 0 ? impl::camb::requiresBuffer(ctx, workspace_size).data() : nullptr;
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transDesc.get(), inDesc.get(),
                                      in.data(), outDesc.get(), out.data(),
                                      workspace_ptr, workspace_size));
    return diopiSuccess;
}

static diopiError_t convertType(cnnlDataType_t *cnnlType, diopiDtype_t type) {
    switch (type) {
    case diopi_dtype_int8:
        *cnnlType = CNNL_DTYPE_INT8;
        break;
    case diopi_dtype_uint8:
        *cnnlType = CNNL_DTYPE_UINT8;
        break;
    case diopi_dtype_int32:
        *cnnlType = CNNL_DTYPE_INT32;
        break;
    case diopi_dtype_uint32:
        *cnnlType = CNNL_DTYPE_UINT32;
        break;
    case diopi_dtype_float16:
        *cnnlType = CNNL_DTYPE_HALF;
        break;
    case diopi_dtype_float32:
        *cnnlType = CNNL_DTYPE_FLOAT;
        break;
    case diopi_dtype_int16:
        *cnnlType = CNNL_DTYPE_INT16;
        break;
    case diopi_dtype_uint16:
        *cnnlType = CNNL_DTYPE_UINT16;
        break;
    case diopi_dtype_bool:
        *cnnlType = CNNL_DTYPE_BOOL;
        break;
    case diopi_dtype_int64:
        *cnnlType = CNNL_DTYPE_INT64;
        break;
    default:
        impl::camb::set_last_error_string("unkown diopitype error %d at %s:%s", type, __FILE__, __LINE__);
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto stream  = impl::camb::getStream(ctx);
    auto trInput = impl::camb::makeTensor(input);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTensorDescriptor_t,
        cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDesc;
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));
    cnnlTensorDescriptor_t desc = CnnlDesc.get();

    diopiSize_t shape = trInput.shape();
    int dimNb = shape.len;
    std::vector<int> dimStrides(dimNb, 1);
    std::vector<int> dimSize(dimNb);
    diopiSize_t stride = trInput.stride();

    if (dimNb == 0) {
        dimNb = 1;
        dimSize.push_back(1);
        dimStrides.push_back(1);
    } else {
        for (int i = 0; i < dimNb; ++i) {
            dimSize[i] = shape.data[i];
        }
        if (dimNb > 0) {
            for (int i = 0; i < dimNb; ++i) {
                dimStrides[i] = stride.data[i];
            }
        }
    }

    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }

    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(desc, layout, dtype, dimNb,
        dimSize.data(), dimStrides.data()));
    DIOPI_CALLCNNL(cnnlFill(handle, val, desc, trInput.data()));
}


diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                      diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                                      diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    /* Generate Tensors */
    auto save_mean_tr = impl::camb::makeTensor(save_mean);
    auto save_invstd_tr = impl::camb::makeTensor(save_invstd);
    auto input_tr = impl::camb::makeTensor(input);
    auto weight_tr = impl::camb::makeTensor(weight);
    auto bias_tr = impl::camb::makeTensor(bias);
    auto running_mean_tr = impl::camb::makeTensor(running_mean);
    auto running_var_tr = impl::camb::makeTensor(running_var);
    auto output_tr = impl::camb::makeTensor(out);

    /* Some basic check */
    DIOPI_CHECK(running_mean_tr.dtype() ==  running_var_tr.dtype(), "running_mean and running_var need to have the same data types");
    // TODO: 2,3,5 dim support
    DIOPI_CHECK(input_tr.dim() >= 4 && input_tr.dim() <=4, "Input dim is out of range");
    DIOPI_CHECK(input_tr.dim() == output_tr.dim(), "Input dim != out dim");

    /* Get current handle*/
    auto stream = impl::camb::getStream(ctx);
    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    /* Transpose NCHW to NHWC */
    auto memory_format = impl::camb::MemoryFormat::ChannelsLast;
    auto input_channel_last = input_tr.contiguous(ctx, memory_format);
    cnnl_transpose(ctx, handle, input_tr, input_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    auto output_channel_last = output_tr.contiguous(ctx, memory_format);
    // cnnl_transpose(ctx, handle, output_tr, output_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

    CnnlTensorDescriptor weight_bias_mean_var_desc(weight_tr, CNNL_LAYOUT_ARRAY);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDescriptor input_channel_last_desc(input_channel_last, layout);
    CnnlTensorDescriptor output_channel_last_desc(output_channel_last, layout);


    //get workspace for BNFT
    size_t workspace_size = 0;
    DIOPI_CHECKCNNL(cnnlGetBatchNormForwardWorkspaceSize(handle, input_channel_last_desc.get(), &workspace_size));

    void* workspace_ptr = workspace_size== 0 ? impl::camb::requiresBuffer(ctx, workspace_size).data() : nullptr;

    // set activition part to default
    cnnlActivationMode_t active_mode = CNNL_ACTIVATION_IDENTITY;
    cnnlActivationDescriptor_t activation_desc = nullptr;
    cnnlCreateActivationDescriptor (&activation_desc);
    cnnlSetActivationDescriptor_v5(activation_desc, active_mode, CNNL_ACTIVATION_HIGH_PRECISION,
                                                        CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);

    if (training) {
        DIOPI_CALLCNNL(cnnlBatchNormForwardTraining_v2(
            /* handle   */ handle,
            /*activation_desc */ activation_desc,
            /*mode */ CNNL_BATCHNORM_SPATIAL,
            /*bnOps */ CNNL_BATCHNORM_OPS_BN,
            /* alpha    */ nullptr,
            /* beta     */ nullptr,
            /* x_desc   */ input_channel_last_desc.get(),
            /* x        */ input_channel_last.data(),
            /* z_desc */ NULL,
            /* z */ NULL,
            /* wbmvd    */ weight_bias_mean_var_desc.get(),
            /* weight   */ weight_tr.data(),
            /* bias     */ bias_tr.data(),
            /* mov_mean */ running_mean_tr.defined() ? running_mean_tr.data() : nullptr,
            /* mov_var  */ running_var_tr.defined() ? running_var_tr.data() : nullptr,
            /* eps      */ static_cast<float>(eps),
            /* momentum */ static_cast<float>(momentum),
            /* y_desc   */ output_channel_last_desc.get(),
            /* y        */ output_channel_last.data(),
            /* save_mean*/ save_mean_tr.data(),
            /* save_std */ save_invstd_tr.data(),
            /* workspace */ workspace_ptr,
            /* workspace_size */ workspace_size,
            /* reservespace */ NULL,
            /* reservespace_size */ 0));
    } else {
        DIOPI_CALLCNNL(cnnlBatchNormForwardInference(
            /* handle   */ handle,
            /* alpha    */ nullptr,
            /* beta     */ nullptr,
            /* x_desc   */ input_channel_last_desc.get(),
            /* x        */ input_channel_last.data(),
            /* wbmvd    */ weight_bias_mean_var_desc.get(),
            /* weight   */ weight_tr.data(),
            /* bias     */ bias_tr.data(),
            /* mov_mean */ running_mean_tr.defined() ? running_mean_tr.data() : nullptr,
            /* mov_var  */ running_var_tr.defined() ? running_var_tr.data() : nullptr,
            /* eps      */ static_cast<float>(eps),
            /* z_desc   */ output_channel_last_desc.get(),
            /* z        */ output_channel_last.data()));
    }

    // NHWC -> NCHW
    cnnl_transpose(ctx, handle, output_channel_last, output_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    // cnrtQueueSync(stream);
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                              diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                              diopiConstTensorHandle_t save_invstd, bool training, double eps){
    /* Generate diopi Stream, Tensors and Handle*/
    auto stream  = impl::camb::getStream(ctx);

    auto grad_input_tr = impl::camb::makeTensor(grad_input);
    auto grad_weight_tr = impl::camb::makeTensor(grad_weight);
    auto grad_bias_tr = impl::camb::makeTensor(grad_bias);
    auto input_tr = impl::camb::makeTensor(input);
    auto weight_tr = impl::camb::makeTensor(weight);
    auto running_mean_tr = impl::camb::makeTensor(running_mean);
    auto running_var_tr = impl::camb::makeTensor(running_var);
    auto save_mean_tr = impl::camb::makeTensor(save_mean);
    auto save_invstd_tr = impl::camb::makeTensor(save_invstd);

    auto grad_out_tr = impl::camb::makeTensor(grad_output);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    /* Some basic check */
    DIOPI_CHECK(running_mean_tr.dtype() ==  running_var_tr.dtype(), "running_mean and running_var need to have the same data types");

    // /* Generate description */
    // cnnlTensorDescriptor weight_bias_mean_var_desc, input_desc, grad_output_desc, grad_input_desc;

    // // When input is 2D or 3D Tensor, Reshape it to 4D tensor
    // // NC -> NC11 -> NCHW | NCD -> NCD1 -> NCHW
    // if (input_dim == 3 || input_dim == 2) {

    //     std::vector<int> input_shape{1, 1, 1, 1};
    //     std::vector<int> grad_out_shape{1, 1, 1, 1};
    //     std::vector<int> grad_input_shape{1, 1, 1, 1};
    //     for (int i = 0; i < input_dim; ++i) {
    //         input_shape[i] = input_tr.shape().data[i];
    //         grad_out_shape[i] = grad_out_tr.shape().data[i];
    //         grad_input_shape[i] = grad_input_tr.shape().data[i];
    //     }
    //     // set cnnl descriptor
    //     input_desc.set_additional_dim(input_tr, input_shape);
    //     grad_output_desc.set_additional_dim(grad_out_tr, grad_out_shape);
    //     grad_input_desc.set_additional_dim(grad_input_tr, grad_input_shape);
    //     weight_bias_mean_var_desc.set(weight_tr, CNNL_LAYOUT_ARRAY);
    // } else {
    //     cnnlTensorLayout_t layout =
    //         input_dim > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
    //     // set cnnl descriptor
    //     input_desc.set(input_tr, layout);
    //     grad_output_desc.set(grad_out_tr, layout);
    //     grad_input_desc.set(grad_input_tr, layout);
    //     weight_bias_mean_var_desc.set(weight_tr, CNNL_LAYOUT_ARRAY);
    // }

    // if (training) {
    //     DIOPI_CALLCNNL(cnnlBatchNormBackward(
    //         /* handle           */ handle,
    //         /* alpha_data_diff  */ nullptr,
    //         /* beta_data_diff   */ nullptr,
    //         /* alpha_param_diff */ nullptr,
    //         /* beta_param_diff  */ nullptr,
    //         /* x_desc           */ input_desc.get(),
    //         /* input                */ input_tr.data(),
    //         /* diff_z_desc      */ grad_output_desc.get(),
    //         /* diff_z           */ grad_out_tr.data(),
    //         /* wbmv_desc        */ weight_bias_mean_var_desc.get(),
    //         /* weight           */ weight_tr.data(),
    //         /* saved_mean       */ save_mean_tr.data(),
    //         /* saved_invstd     */ save_invstd_tr.data(),
    //         /* eps              */ static_cast<float>(eps),
    //         /* grad_input_desc    */ grad_input_desc.get(),
    //         /* diff_x           */ grad_input_tr.data(),
    //         /* diff_weight      */ grad_weight_tr.data(),
    //         /* diff_bias        */ grad_bias_tr.data()));
    // } else {
    //     DIOPI_CALLCNNL(cnnlFrozenBatchNormBackward(
    //         /* handle       */  handle,
    //         /* x_desc       */  input_desc.get(),
    //         /* input            */  input_tr.data(),
    //         /* diff_y_desc  */  grad_output_desc.get(),
    //         /* diff_y       */  grad_out_tr.data(),
    //         /* wbmv_desc    */  weight_bias_mean_var_desc.get(),
    //         /* weight       */  weight_tr.data(),
    //         /* pop_mean     */  running_mean_tr.data(),
    //         /* pop_var      */  running_var_tr.data(),
    //         /* eps          */  static_cast<float>(eps),
    //         /* grad_input_desc */ grad_input_desc.get(),
    //         /* diff_x        */ grad_input_tr.data(),
    //         /* diff_weight   */ grad_weight_tr.data(),
    //         /* diff_bias     */ grad_bias_tr.data()));
    // }
    // cnrtQueueSync(stream);
    return diopiSuccess;
}

}  // extern "C"

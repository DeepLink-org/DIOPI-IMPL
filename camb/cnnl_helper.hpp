#ifndef CNNL_HELPER_HPP_
#define CNNL_HELPER_HPP_

#include <cnnl.h>
#include "diopi_helper.hpp"
#include <vector>

#define DIOPI_CHECK(cond, str)                                             \
    do {                                                                   \
        if (!(cond)) {                                                     \
            impl::camb::set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                     \
        }                                                                  \
    } while (false);

#define DIOPI_CALLCNNL(Expr)                                                 \
    do {                                                                     \
        ::cnnlStatus_t ret = Expr;                                           \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                  \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s", \
                                              ret,                           \
                                              ::cnnlGetErrorString(ret),     \
                                              __FILE__,                      \
                                              __LINE__);                     \
            return diopiErrorOccurred;                                       \
        }                                                                    \
    } while (false);

#define DIOPI_CHECKCNNL(Expr)                                                \
    do {                                                                     \
        ::cnnlStatus_t ret = Expr;                                           \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                  \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s", \
                                              ret,                           \
                                              ::cnnlGetErrorString(ret),     \
                                              __FILE__,                      \
                                              __LINE__);                     \
        }                                                                    \
    } while (false);

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


diopiError_t convertType(cnnlDataType_t *cnnlType, diopiDtype_t type);

template<typename T>
diopiError_t cnnl_transpose(diopiContextHandle_t& ctx, cnnlHandle_t& handle, T& in, impl::camb::DiopiTensor<diopiTensorHandle_t>& out, cnnlTensorLayout_t layoutIn,
                          cnnlTensorLayout_t layoutOut){
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

#endif  // CNNL_HELPER_HPP_
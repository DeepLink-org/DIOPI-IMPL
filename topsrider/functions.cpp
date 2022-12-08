/*
* SPDX-FileCopyrightText: Copyright (c) 2022 Enflame. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
#include <diopi/functions.h>
#include <tops_runtime.h>
#include <topsdnn.h>
#include <cstdio>
#include <iostream>
#include <vector>

#include "helper.hpp"
#define DIOPI_CALLTOPS(Expr)                                              \
  {                                                                       \
    topsError_t ret = Expr;                                               \
    if (ret != topsSuccess) {                                             \
      printf("call a topsrt function (%s) failed. return code=%d", #Expr, \
             ret);                                                        \
      return diopiErrorOccurred;                                          \
    }                                                                     \
  }

#define DIOPI_CALLTOPSDNN(Expr)                                                \
  {                                                                            \
    ::topsdnnStatus_t ret = Expr;                                              \
    if (TOPSDNN_STATUS_SUCCESS != ret) {                                       \
      impl::tops::set_last_error_string("topsdnn error %d : %s at %s:%s", ret, \
                                        "topsGetErrorString(ret)", __FILE__,   \
                                        __LINE__);                             \
      return diopiErrorOccurred;                                               \
    }                                                                          \
  }

#define DIOPI_CHECKTOPSDNN(Expr)                                               \
  {                                                                            \
    ::topsdnnStatus_t ret = Expr;                                              \
    if (TOPSDNN_STATUS_SUCCESS != ret) {                                       \
      impl::tops::set_last_error_string("topsdnn error %d : %s at %s:%s", ret, \
                                        "topsGetErrorString(ret)", __FILE__,   \
                                        __LINE__);                             \
    }                                                                          \
  }

static diopiError_t convertType(topsdnnDataType_t* topsdnnType,
                                diopiDtype_t type) {
  switch (type) {
    case diopi_dtype_int8:
      *topsdnnType = TOPSDNN_DATA_INT8;
      break;
    // case diopi_dtype_uint8:
    //     *topsdnnType = TOPSDNN_DATA_UINT8;
    //     break;
    case diopi_dtype_int32:
      *topsdnnType = TOPSDNN_DATA_INT32;
      break;
    case diopi_dtype_float16:
      *topsdnnType = TOPSDNN_DATA_HALF;
      break;
    case diopi_dtype_float32:
      *topsdnnType = TOPSDNN_DATA_FLOAT;
      break;
    case diopi_dtype_float64:
      *topsdnnType = TOPSDNN_DATA_DOUBLE;
      break;
    default:
      impl::tops::set_last_error_string("unkown diopitype error %d at %s:%s",
                                        type, __FILE__, __LINE__);
      return diopiDtypeNotSupported;
  }
  return diopiSuccess;
}

namespace impl {

namespace tops {

class TopsdnnScalar final {
 public:
  template <typename T>
  void reset(const T& val) {
    if (data_) delete[] data_;
    data_ = new int8_t[sizeof(T)];
    T* ptr = reinterpret_cast<T*>(data_);
    *ptr = val;
  }

  void* data() const { return data_; }

  ~TopsdnnScalar() {
    if (data_) delete[] data_;
  }

 protected:
  int8_t* data_{nullptr};
};

template <typename T, topsdnnStatus_t (*fnCreate)(T*),
          topsdnnStatus_t (*fnDestroy)(T)>
class TopsdnnResourceGuard final {
 public:
  TopsdnnResourceGuard() { DIOPI_CHECKTOPSDNN(fnCreate(&resource_)); }

  ~TopsdnnResourceGuard() { DIOPI_CHECKTOPSDNN(fnDestroy(resource_)); }

  T& get() { return resource_; }

 protected:
  T resource_{0};
};

diopiError_t setTensorDesc(diopiDtype_t type, const diopiSize_t& shape,
                           const diopiSize_t& stride,
                           topsdnnTensorDescriptor_t desc) {
  topsdnnDataType_t topsdnnType;
  DIOPI_CALL(convertType(&topsdnnType, type));

  int len = shape.len;
  int size = len < 4 ? 4 : len;
  std::vector<int> shapeArray(size);
  std::vector<int> strideArray(size);

  for (int i = 0; i < len; ++i) {
    shapeArray[i] = shape.data[i];
    strideArray[i] = stride.data[i];
  }
  for (int i = len; i < 4; ++i) {
    shapeArray[i] = 1;
    strideArray[i] = 1;
  }

  DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(
      desc, topsdnnType, size, shapeArray.data(), strideArray.data()));
  return diopiSuccess;
}

diopiError_t topsTranspose(topsdnnHandle_t handle, void* input, void* output,
                           int n, int c, int h, int w,
                           topsdnnDataType_t dataType,
                           topsdnnTensorFormat_t inFormat) {
  float alpha = 1.0f;
  float beta = 0.0f;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      in_desc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      out_desc;

  int in_size = n * h * w * c;
  int dims[4] = {n, c, h, w};
  int nhwc_strides[4] = {h * w * c, 1, w * c, c};
  int nchw_strides[4] = {c * h * w, h * w, w, 1};

  // Set descriptors
  if (inFormat == TOPSDNN_TENSOR_NCHW) {
    DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(in_desc.get(), dataType, 4,
                                                   dims, nchw_strides));
    DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(out_desc.get(), dataType, 4,
                                                   dims, nhwc_strides));
  } else {
    DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(in_desc.get(), dataType, 4,
                                                   dims, nhwc_strides));
    DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(out_desc.get(), dataType, 4,
                                                   dims, nchw_strides));
  }

  DIOPI_CALLTOPSDNN(topsdnnTransformTensor(handle, &alpha, in_desc.get(), input,
                                           &beta, out_desc.get(), output));
  return diopiSuccess;
}

}  // namespace tops

}  // namespace impl

extern "C" {

static const char* name = "GcuDevice";
static char version[1024] = {0};

const char* diopiGetVendorName() { return name; }

const char* diopiGetImplVersion() {
  int rt_version = 2009;
  if (strlen(version) == 0) {
    const char* diopiVersion = diopiGetVersion();
    sprintf(version, "TopsRt Version: %d; Topsdnn Version: %d; %s", rt_version,
            TOPSDNN_VERSION, diopiVersion);
  }
  return version;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                          diopiConstTensorHandle_t input, int64_t dim,
                          diopiDtype_t dtype) {
  if (dim > 1) {
    impl::tops::set_last_error_string("unkown dim error dim=%d at %s:%s", dim,
                                      __FILE__, __LINE__);
    return diopiErrorOccurred;
  }

  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      desc;

  auto trIn = impl::tops::makeTensor(input);
  auto trOut = impl::tops::makeTensor(out);
  auto stream = impl::tops::getStream(ctx);
  if (0 == dim) {
    diopiSize_t oldShape = trIn.shape();
    diopiSize_t oldStride = trIn.stride();
    diopiSize_t newShape, newStride;
    int64_t len = oldShape.len + 1;
    std::vector<int64_t> shape(len);
    std::vector<int64_t> stride(len);
    shape[0] = 1;
    stride[0] = oldStride.data[0];
    for (int i = 0; i < oldShape.len; ++i) {
      shape[i + 1] = oldShape.data[i];
      stride[i + 1] = oldStride.data[i];
    }
    newShape.data = shape.data();
    newShape.len = len;
    newStride.data = stride.data();
    newStride.len = len;
    DIOPI_CALL(impl::tops::setTensorDesc(trIn.dtype(), newShape, newStride,
                                         desc.get()));
  } else {
    DIOPI_CALL(impl::tops::setTensorDesc(trIn.dtype(), trIn.shape(),
                                         trIn.stride(), desc.get()));
  }

  impl::tops::TopsdnnScalar alpha, beta;
  if (dtype == diopi_dtype_float64) {
    alpha.reset<double>(1.0);
    beta.reset<double>(0.0);
  } else {
    alpha.reset<float>(1.f);
    beta.reset<float>(0.f);
  }
  // DIOPI_CALLTOPSDNN(topsdnnSetStream(handle.get(), stream));
  DIOPI_CALLTOPSDNN(topsdnnSoftmaxForward(
      handle.get(), TOPSDNN_SOFTMAX_ACCURATE, TOPSDNN_SOFTMAX_MODE_CHANNEL,
      alpha.data(), desc.get(), trIn.data(), beta.data(), desc.get(),
      trOut.data()));

  return diopiSuccess;
}

diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                       diopiConstTensorHandle_t input) {
  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      desc;
  impl::tops::TopsdnnResourceGuard<topsdnnActivationDescriptor_t,
                                   topsdnnCreateActivationDescriptor,
                                   topsdnnDestroyActivationDescriptor>
      descAct;

  auto trIn = impl::tops::makeTensor(input);
  auto trOut = impl::tops::makeTensor(out);
  // auto stream = impl::tops::getStream(ctx);

  DIOPI_CALL(impl::tops::setTensorDesc(trIn.dtype(), trIn.shape(),
                                       trIn.stride(), desc.get()));
  DIOPI_CALLTOPSDNN(topsdnnSetActivationDescriptor(
      descAct.get(), TOPSDNN_ACTIVATION_RELU, TOPSDNN_PROPAGATE_NAN, 0.0));

  impl::tops::TopsdnnScalar alpha, beta;
  if (trIn.dtype() == diopi_dtype_float64) {
    alpha.reset<double>(1.0);
    beta.reset<double>(0.0);
  } else {
    alpha.reset<float>(1.f);
    beta.reset<float>(0.f);
  }
  // DIOPI_CALLTOPSDNN(topsdnnSetStream(handle.get(), stream));
  DIOPI_CALLTOPSDNN(topsdnnActivationForward(
      handle.get(), descAct.get(), alpha.data(), desc.get(), trIn.data(),
      beta.data(), desc.get(), trOut.data()));

  return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      desc;
  impl::tops::TopsdnnResourceGuard<topsdnnActivationDescriptor_t,
                                   topsdnnCreateActivationDescriptor,
                                   topsdnnDestroyActivationDescriptor>
      descAct;

  auto trIn = impl::tops::makeTensor(input);
  //   auto stream = impl::tops::getStream(ctx);

  DIOPI_CALL(impl::tops::setTensorDesc(trIn.dtype(), trIn.shape(),
                                       trIn.stride(), desc.get()));
  DIOPI_CALLTOPSDNN(topsdnnSetActivationDescriptor(
      descAct.get(), TOPSDNN_ACTIVATION_RELU, TOPSDNN_PROPAGATE_NAN, 0.0));

  impl::tops::TopsdnnScalar alpha, beta;
  if (trIn.dtype() == diopi_dtype_float64) {
    alpha.reset<double>(1.0);
    beta.reset<double>(0.0);
  } else {
    alpha.reset<float>(1.f);
    beta.reset<float>(0.f);
  }
  // DIOPI_CALLTOPSDNN(topsdnnSetStream(handle.get(), stream));
  DIOPI_CALLTOPSDNN(topsdnnActivationForward(
      handle.get(), descAct.get(), alpha.data(), desc.get(), trIn.data(),
      beta.data(), desc.get(), trIn.data()));

  return diopiSuccess;
}

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx,
                                diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias,
                                diopiSize_t stride, diopiSize_t padding,
                                diopiSize_t dilation, int64_t groups) {
  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      descInput;
  impl::tops::TopsdnnResourceGuard<topsdnnFilterDescriptor_t,
                                   topsdnnCreateFilterDescriptor,
                                   topsdnnDestroyFilterDescriptor>
      descW;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      descOut;
  impl::tops::TopsdnnResourceGuard<topsdnnConvolutionDescriptor_t,
                                   topsdnnCreateConvolutionDescriptor,
                                   topsdnnDestroyConvolutionDescriptor>
      descConv;
  topsdnnConvolutionMode_t mode = TOPSDNN_CROSS_CORRELATION;
  float alpha = 1.f;
  float beta = 0.f;
  int out_h, out_w, out_c, out_n;
  int in_h, in_w, in_c, in_n;
  size_t fwd_workspace_size;
  void* fwd_workspace = NULL;
  void* dtuDevPtrI = NULL;
  void* dtuDevPtrF = NULL;
  void* dtuDevPtrO = NULL;
  topsdnnConvolutionFwdAlgo_t fwd_algo;

  auto trIn = impl::tops::makeTensor(input);
  auto trW = impl::tops::makeTensor(weight);
  auto trB = impl::tops::makeTensor(bias);
  auto trOut = impl::tops::makeTensor(out);

  in_n = trIn.shape().data[0];
  in_c = trIn.shape().data[1];
  in_h = trIn.shape().data[2];
  in_w = trIn.shape().data[3];

  topsdnnDataType_t topsdnnType;
  DIOPI_CALL(convertType(&topsdnnType, trIn.dtype()));
  int input_dim[] = {in_n, in_c, in_h, in_w};
  int input_stride[] = {in_c * in_h * in_w, 1, in_c * in_h, in_c};
  DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(descInput.get(), topsdnnType,
                                                 4, input_dim, input_stride));
  // create conv desc
  int pad_array[] = {(int)padding.data[0], (int)padding.data[1]};
  int stride_array[] = {(int)stride.data[0], (int)stride.data[1]};
  int dilation_array[] = {(int)dilation.data[0], (int)dilation.data[1]};
  DIOPI_CALLTOPSDNN(topsdnnSetConvolutionNdDescriptor(
      descConv.get(), 2, pad_array, stride_array, dilation_array, mode,
      topsdnnType));

  int filter_dim[] = {(int)trW.shape().data[0], (int)trW.shape().data[1],
                      (int)trW.shape().data[2], (int)trW.shape().data[3]};
  DIOPI_CALLTOPSDNN(topsdnnSetFilterNdDescriptor(
      descW.get(), topsdnnType, TOPSDNN_TENSOR_NHWC, 4, filter_dim));
  DIOPI_CALLTOPSDNN(topsdnnGetConvolution2dForwardOutputDim(
      descConv.get(), descInput.get(), descW.get(), &out_n, &out_c, &out_h,
      &out_w));

  int out_dim[] = {out_n, out_c, out_h, out_w};
  int out_stride[] = {out_c * out_h * out_w, 1, out_c * out_w, out_c};
  DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(descOut.get(), topsdnnType, 4,
                                                 out_dim, out_stride));
  int returnedAlgoCount;
  int requestedAlgoCount = 8;
  topsdnnConvolutionFwdAlgoPerf_t perfResults[8];
  DIOPI_CALLTOPSDNN(topsdnnFindConvolutionForwardAlgorithm(
      handle.get(), descInput.get(), descW.get(), descConv.get(), descOut.get(),
      requestedAlgoCount, &returnedAlgoCount, perfResults));
  // Choose forward algorithm
  fwd_algo = perfResults[0].algo;
  // Set workspace size
  DIOPI_CALLTOPSDNN(topsdnnGetConvolutionForwardWorkspaceSize(
      handle.get(), descInput.get(), descW.get(), descConv.get(), descOut.get(),
      fwd_algo, &fwd_workspace_size));

  // Allocate device memory
  DIOPI_CALLTOPS(topsMalloc(reinterpret_cast<void**>(&dtuDevPtrI),
                            sizeof(float) * trIn.shape().data[0] *
                                trIn.shape().data[1] * trIn.shape().data[2] *
                                trIn.shape().data[3]));
  DIOPI_CALLTOPS(topsMalloc(reinterpret_cast<void**>(&dtuDevPtrF),
                            sizeof(float) * trW.shape().data[0] *
                                trW.shape().data[1] * trW.shape().data[2] *
                                trW.shape().data[3]));
  DIOPI_CALLTOPS(topsMalloc(reinterpret_cast<void**>(&dtuDevPtrO),
                            sizeof(float) * out_n * out_c * out_h * out_w));
  DIOPI_CALL(impl::tops::topsTranspose(
      handle.get(), (void*)trIn.data(), (void*)dtuDevPtrI, trIn.shape().data[0],
      trIn.shape().data[1], trIn.shape().data[2], trIn.shape().data[3],
      topsdnnType, TOPSDNN_TENSOR_NCHW));
  DIOPI_CALL(impl::tops::topsTranspose(
      handle.get(), (void*)trW.data(), (void*)dtuDevPtrF, trW.shape().data[0],
      trW.shape().data[1], trW.shape().data[2], trW.shape().data[3],
      topsdnnType, TOPSDNN_TENSOR_NCHW));
  DIOPI_CALLTOPSDNN(topsdnnConvolutionForward(
      handle.get(), &alpha, descInput.get(), dtuDevPtrI, descW.get(),
      dtuDevPtrF, descConv.get(), fwd_algo, fwd_workspace, fwd_workspace_size,
      &beta, descOut.get(), dtuDevPtrO));
  DIOPI_CALL(impl::tops::topsTranspose(
      handle.get(), (void*)dtuDevPtrO, (void*)trOut.data(), out_n, out_c, out_h,
      out_w, topsdnnType, TOPSDNN_TENSOR_NHWC));

  if (bias) {
    float alpha = 1.0f;
    float beta = 1.0f;
    int dimB_padded[4];
    dimB_padded[0] = 1;
    dimB_padded[1] = out_c;
    dimB_padded[2] = 1;
    dimB_padded[3] = 1;
    int strideB_padded[4] = {out_c, 1, 1, 1};
    impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                     topsdnnCreateTensorDescriptor,
                                     topsdnnDestroyTensorDescriptor>
        biasDesc;
    impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                     topsdnnCreateTensorDescriptor,
                                     topsdnnDestroyTensorDescriptor>
        h_desc;
    DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(
        biasDesc.get(), topsdnnType, 4, dimB_padded, strideB_padded));
    int out_stride[] = {out_c * out_h * out_w, out_h * out_w, out_w, 1};
    DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(h_desc.get(), topsdnnType, 4,
                                                   out_dim, out_stride));
    DIOPI_CALLTOPSDNN(topsdnnAddTensor(handle.get(), &alpha, biasDesc.get(),
                                       (void*)trB.data(), &beta, h_desc.get(),
                                       (void*)trOut.data()));
  }
  DIOPI_CALLTOPS(topsFree(dtuDevPtrI));
  DIOPI_CALLTOPS(topsFree(dtuDevPtrO));
  DIOPI_CALLTOPS(topsFree(dtuDevPtrF));
  return diopiSuccess;
}

diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                            diopiConstTensorHandle_t input,
                            diopiSize_t kernel_size, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation,
                            bool ceil_mode) {
  if (ceil_mode) {
    impl::tops::set_last_error_string("not support ceil_mode at %s:%s",
                                      __FILE__, __LINE__);
    return diopiErrorOccurred;
  }

  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      x_desc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      h_desc;
  impl::tops::TopsdnnResourceGuard<topsdnnPoolingDescriptor_t,
                                   topsdnnCreatePoolingDescriptor,
                                   topsdnnDestroyPoolingDescriptor>
      pooling_desc;
  int insize = 0;
  int outsize = 0;
  auto trIn = impl::tops::makeTensor(input);
  auto trOut = impl::tops::makeTensor(out);

  const float alpha = 1.f;
  const float beta = 0.f;
  int window_w = kernel_size.data[0];
  int window_h = kernel_size.data[1];
  int v_padding = padding.data[0];
  int h_padding = padding.data[1];
  int v_stride = stride.data[0];
  int h_stride = stride.data[1];

  int n = trIn.shape().data[0];
  int c = trIn.shape().data[1];
  int h = trIn.shape().data[2];
  int w = trIn.shape().data[3];

  topsdnnDataType_t topsdnnType;
  DIOPI_CALL(convertType(&topsdnnType, trIn.dtype()));

  DIOPI_CALLTOPSDNN(topsdnnSetPooling2dDescriptor(
      pooling_desc.get(),     // descriptor handle
      TOPSDNN_POOLING_MAX,    // mode - max pooling
      TOPSDNN_PROPAGATE_NAN,  // NaN propagation mode
      window_h,               // window height
      window_w,               // window width
      v_padding,              // vertical padding
      h_padding,              // horizontal padding
      v_stride,               // vertical stride
      h_stride));             // horizontal stride

  DIOPI_CALLTOPSDNN(topsdnnSetTensor4dDescriptor(
      x_desc.get(), TOPSDNN_TENSOR_NHWC, topsdnnType, n, c, h, w));

  int outN = 1, outC = 1, outH = 1, outW = 1;
  DIOPI_CALLTOPSDNN(topsdnnGetPooling2dForwardOutputDim(
      pooling_desc.get(), x_desc.get(), &outN, &outC, &outH, &outW));

  DIOPI_CALLTOPSDNN(topsdnnSetTensor4dDescriptor(
      h_desc.get(), TOPSDNN_TENSOR_NHWC, topsdnnType, outN, outC, outH, outW));

  insize = n * c * h * w;
  outsize = outN * outC * outH * outW;

  // to-do: support other dtype
  float* dtuDevPtrI = NULL;
  float* dtuDevPtrO = NULL;
  // Allocate device memory
  DIOPI_CALLTOPS(topsMalloc(reinterpret_cast<void**>(&dtuDevPtrI),
                            sizeof(float) * insize));
  DIOPI_CALLTOPS(topsMalloc(reinterpret_cast<void**>(&dtuDevPtrO),
                            sizeof(float) * outsize));

  DIOPI_CALL(impl::tops::topsTranspose(handle.get(), (void*)trIn.data(),
                                       (void*)dtuDevPtrI, n, c, h, w,
                                       topsdnnType, TOPSDNN_TENSOR_NCHW));

  // Pooling forward
  DIOPI_CALLTOPSDNN(
      topsdnnPoolingForward(handle.get(),        // context handle
                            pooling_desc.get(),  // pooling descriptor
                            &alpha,              // alpha scaling factor
                            x_desc.get(),        // input tensor descriptor
                            dtuDevPtrI,          // input data pointer
                            &beta,               // beta scaling factor
                            h_desc.get(),        // output tensor descriptor
                            dtuDevPtrO));        // output data pointer
  DIOPI_CALL(impl::tops::topsTranspose(handle.get(), (void*)dtuDevPtrO,
                                       (void*)trOut.data(), outN, outC, outH,
                                       outW, topsdnnType, TOPSDNN_TENSOR_NHWC));

  DIOPI_CALLTOPS(topsFree(dtuDevPtrI));
  DIOPI_CALLTOPS(topsFree(dtuDevPtrO));
  return diopiSuccess;
}

diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                            diopiConstTensorHandle_t input,
                            diopiSize_t kernel_size, diopiSize_t stride,
                            diopiSize_t padding, bool ceil_mode,
                            bool count_include_pad,
                            const int64_t* divisor_override) {
  if (ceil_mode) {
    impl::tops::set_last_error_string("not support ceil_mode at %s:%s",
                                      __FILE__, __LINE__);
    return diopiErrorOccurred;
  }

  if (divisor_override != nullptr) {
    impl::tops::set_last_error_string("not support divisor_override at %s:%s",
                                      __FILE__, __LINE__);
    return diopiErrorOccurred;
  }

  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      x_desc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      h_desc;
  impl::tops::TopsdnnResourceGuard<topsdnnPoolingDescriptor_t,
                                   topsdnnCreatePoolingDescriptor,
                                   topsdnnDestroyPoolingDescriptor>
      pooling_desc;
  int insize = 0;
  int outsize = 0;
  auto trIn = impl::tops::makeTensor(input);
  auto trOut = impl::tops::makeTensor(out);

  const float alpha = 1.f;
  const float beta = 0.f;
  int window_w = kernel_size.data[0];
  int window_h = kernel_size.data[1];
  int v_padding = padding.data[0];
  int h_padding = padding.data[1];
  int v_stride = stride.data[0];
  int h_stride = stride.data[1];

  int n = trIn.shape().data[0];
  int c = trIn.shape().data[1];
  int h = trIn.shape().data[2];
  int w = trIn.shape().data[3];

  topsdnnDataType_t topsdnnType;
  topsdnnPoolingMode_t pooling_mode;
  DIOPI_CALL(convertType(&topsdnnType, trIn.dtype()));

  if (count_include_pad) {
    pooling_mode = TOPSDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else {
    pooling_mode = TOPSDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }

  DIOPI_CALLTOPSDNN(topsdnnSetPooling2dDescriptor(
      pooling_desc.get(),     // descriptor handle
      pooling_mode,           // mode - max pooling
      TOPSDNN_PROPAGATE_NAN,  // NaN propagation mode
      window_h,               // window height
      window_w,               // window width
      v_padding,              // vertical padding
      h_padding,              // horizontal padding
      v_stride,               // vertical stride
      h_stride));             // horizontal stride

  DIOPI_CALLTOPSDNN(topsdnnSetTensor4dDescriptor(
      x_desc.get(), TOPSDNN_TENSOR_NHWC, topsdnnType, n, c, h, w));

  int outN = 1, outC = 1, outH = 1, outW = 1;
  DIOPI_CALLTOPSDNN(topsdnnGetPooling2dForwardOutputDim(
      pooling_desc.get(), x_desc.get(), &outN, &outC, &outH, &outW));

  DIOPI_CALLTOPSDNN(topsdnnSetTensor4dDescriptor(
      h_desc.get(), TOPSDNN_TENSOR_NHWC, topsdnnType, outN, outC, outH, outW));

  insize = n * c * h * w;
  outsize = outN * outC * outH * outW;

  // to-do: support other dtype
  float* dtuDevPtrI = NULL;
  float* dtuDevPtrO = NULL;
  // Allocate device memory
  DIOPI_CALLTOPS(topsMalloc(reinterpret_cast<void**>(&dtuDevPtrI),
                            sizeof(float) * insize));
  DIOPI_CALLTOPS(topsMalloc(reinterpret_cast<void**>(&dtuDevPtrO),
                            sizeof(float) * outsize));

  DIOPI_CALL(impl::tops::topsTranspose(handle.get(), (void*)trIn.data(),
                                       (void*)dtuDevPtrI, n, c, h, w,
                                       topsdnnType, TOPSDNN_TENSOR_NCHW));

  // Pooling forward
  DIOPI_CALLTOPSDNN(
      topsdnnPoolingForward(handle.get(),        // context handle
                            pooling_desc.get(),  // pooling descriptor
                            &alpha,              // alpha scaling factor
                            x_desc.get(),        // input tensor descriptor
                            dtuDevPtrI,          // input data pointer
                            &beta,               // beta scaling factor
                            h_desc.get(),        // output tensor descriptor
                            dtuDevPtrO));        // output data pointer
  DIOPI_CALL(impl::tops::topsTranspose(handle.get(), (void*)dtuDevPtrO,
                                       (void*)trOut.data(), outN, outC, outH,
                                       outW, topsdnnType, TOPSDNN_TENSOR_NHWC));

  DIOPI_CALLTOPS(topsFree(dtuDevPtrI));
  DIOPI_CALLTOPS(topsFree(dtuDevPtrO));
  return diopiSuccess;
}

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                         diopiConstTensorHandle_t input,
                         diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;

  auto trIn = impl::tops::makeTensor(input);
  auto trW = impl::tops::makeTensor(weight);
  auto trB = impl::tops::makeTensor(bias);
  auto trOut = impl::tops::makeTensor(out);

  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      aDesc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      bDesc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      cDesc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      oDesc;
  impl::tops::TopsdnnResourceGuard<topsdnnDotDescriptor_t,
                                   topsdnnCreateDotDescriptor,
                                   topsdnnDestroyDotDescriptor>
      dotDesc;

  topsdnnDataType_t topsdnnType;
  DIOPI_CALL(convertType(&topsdnnType, trIn.dtype()));

  float alpha = 1.0f;
  float beta = 0.0f;
  int m = trIn.shape().data[0];
  int k = trIn.shape().data[1];
  int n = trW.shape().data[0];

  int aDescDim[] = {m, k};
  int aStride[] = {k, 1};
  DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(aDesc.get(), topsdnnType, 2,
                                                 aDescDim, aStride));

  int bDescDim[] = {n, k};
  int bStride[] = {k, 1};
  DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(bDesc.get(), topsdnnType, 2,
                                                 bDescDim, bStride));

  int cDescDim[] = {m, n};
  int cStride[] = {n, 1};
  DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(cDesc.get(), topsdnnType, 2,
                                                 cDescDim, cStride));

  int oDescDim[] = {1, n};
  int oStride[] = {n, 1};
  DIOPI_CALLTOPSDNN(topsdnnSetTensorNdDescriptor(oDesc.get(), topsdnnType, 2,
                                                 oDescDim, oStride));

  DIOPI_CALLTOPSDNN(topsdnnSetDotDescriptor(dotDesc.get(), 1, 1, 0, NULL));
  // Do the actual multiplication
  DIOPI_CALLTOPSDNN(topsdnnDot(handle.get(), &alpha, aDesc.get(),
                               (void*)trIn.data(), bDesc.get(),
                               (void*)trW.data(), dotDesc.get(), &beta,
                               cDesc.get(), (void*)trOut.data()));

  if (bias) {
    beta = 1.0f;
    DIOPI_CALLTOPSDNN(topsdnnAddTensor(handle.get(), &alpha, oDesc.get(),
                                       (void*)trB.data(), &beta, cDesc.get(),
                                       (void*)trOut.data()));
  }
  return diopiSuccess;
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                      diopiConstTensorHandle_t input,
                      diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
  impl::tops::TopsdnnResourceGuard<topsdnnHandle_t, topsdnnCreate,
                                   topsdnnDestroy>
      handle;

  topsdnnOpTensorOp_t op = TOPSDNN_OP_TENSOR_ADD;

  auto trIn = impl::tops::makeTensor(input);
  auto trOther = impl::tops::makeTensor(other);
  auto trOut = impl::tops::makeTensor(out);

  // to-do: get alpha value
  float alpha1 = 1.0f;
  float alpha2 = 1.0f;
  float beta = 0.0f;

  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      aDesc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      bDesc;
  impl::tops::TopsdnnResourceGuard<topsdnnTensorDescriptor_t,
                                   topsdnnCreateTensorDescriptor,
                                   topsdnnDestroyTensorDescriptor>
      cDesc;

  impl::tops::TopsdnnResourceGuard<topsdnnOpTensorDescriptor_t,
                                   topsdnnCreateOpTensorDescriptor,
                                   topsdnnDestroyOpTensorDescriptor>
      opDesc;

  DIOPI_CALL(impl::tops::setTensorDesc(trIn.dtype(), trIn.shape(),
                                       trIn.stride(), aDesc.get()));
  DIOPI_CALL(impl::tops::setTensorDesc(trOther.dtype(), trOther.shape(),
                                       trOther.stride(), bDesc.get()));
  DIOPI_CALL(impl::tops::setTensorDesc(trOut.dtype(), trOut.shape(),
                                       trOut.stride(), cDesc.get()));

  DIOPI_CALLTOPSDNN(topsdnnSetOpTensorDescriptor(
      opDesc.get(), op, TOPSDNN_DATA_FLOAT, TOPSDNN_NOT_PROPAGATE_NAN));

  DIOPI_CALLTOPSDNN(topsdnnOpTensor(handle.get(), opDesc.get(), &alpha1,
                                    aDesc.get(), (void*)trIn.data(), &alpha2,
                                    bDesc.get(), (void*)trOther.data(), &beta,
                                    cDesc.get(), (void*)trOut.data()));

  return diopiSuccess;
}

}  // extern "C"
#include "../helper.hpp"
#include <memory>

extern "C" diopiError_t diopiSoftmax(diopiContextHandle_t ctx,
                                               diopiTensorHandle_t out,
                                               diopiConstTensorHandle_t input,
                                               int64_t dim,
                                               diopiDtype_t dtype) {
    HandleGuard handle_guard;
    auto stream = impl::camb::getStream(ctx);
    cnnlHandle_t handle = handle_guard.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    auto input_tensor  = impl::camb::makeTensor(input);
    auto output_tensor = impl::camb::makeTensor(out);

    std::vector<int> src_input_shape  = input_tensor.shape();
    std::vector<int> src_output_shape = output_tensor.shape(); 
    
    const int input_rank = input_tensor.shape_len();
    int mode = dim;
    mode = (mode < 0) ? (mode + input_rank) : mode;
    const size_t input_dim = 3;
    std::vector<int> input_shape(input_dim, 1);
    if (input_rank != 0) {
        if (input_rank <= 3) {
            input_shape[2] = src_input_shape[input_rank - 1];
            input_shape[1] =
                (input_rank == 1) ? 1 : src_input_shape[input_rank - 2];
            input_shape[0] = (input_rank == 3) ? src_input_shape[0] : 1;
        } else {
            auto reduce_dim =
                [](const std::vector<int>& data, int from, int to) -> int {
                to = std::min<int>(to, data.size());
                from = std::max<int>(0, from);
                return std::accumulate(data.cbegin() + from,
                                       data.cbegin() + to + 1,
                                       1LL,
                                       std::multiplies<int64_t>());
            };
            const bool flag = (mode == input_rank - 1);
            input_shape[0] =
                reduce_dim(src_input_shape, 0, flag ? (mode - 2) : (mode - 1));
            input_shape[1] = src_input_shape[flag ? (mode - 1) : mode];
            input_shape[2] = reduce_dim(
                src_input_shape, flag ? mode : (mode + 1), (input_rank - 1));
        }
    }
    cnnlSoftmaxMode_t mode_;
    if (input_rank == 3 && mode == 0) {
        mode_ = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
    } else if (mode == input_rank - 1) {
        mode_ = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    } else {
        mode_ = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    }

    cnnlDataType_t input_dtype;
    DIOPI_CALL(convertType(&input_dtype, input_tensor.dtype()));
    if (CNNL_DTYPE_DOUBLE == input_dtype) {
        return diopiDtypeNotSupported;
    }

    cnnlDataType_t out_dtype;
    DIOPI_CALL(convertType(&out_dtype, dtype));

    TensorDescGuard x_desc_guard, y_desc_guard;
    auto x_desc = x_desc_guard.get();
    auto y_desc = y_desc_guard.get();

    const float alpha = 1;
    const float beta = 0;

    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(
        x_desc, CNNL_LAYOUT_ARRAY, input_dtype, input_dim, input_shape.data()));
    DIOPI_CALLCNNL(cnnlSetTensorDescriptor(
        y_desc, CNNL_LAYOUT_ARRAY, out_dtype, input_dim, input_shape.data()));

    DIOPI_CALLCNNL(cnnlSoftmaxForward_v2(handle,
                                         CNNL_SOFTMAX_ACCURATE,
                                         mode_,
                                         CNNL_COMPUTATION_FAST,
                                         &alpha,
                                         x_desc,
                                         input_tensor.data(),
                                         &beta,
                                         y_desc,
                                         output_tensor.data()));
}

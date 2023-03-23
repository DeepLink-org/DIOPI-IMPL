#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

template <typename T>
void PrintVec(std::vector<T> vec) {
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << std::endl;
    }
}
extern "C" {

DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    for (int i = 0; i < numTensors; i++) {
        auto temp_tensor = DiopiTensor(tensors[i]);
        std::vector<int64_t> temp_tensor_shape = temp_tensor.shape();
        std::vector<int64_t> cat_shape = temp_tensor_shape;
        cat_shape.insert(cat_shape.begin(), 1);
        std::cout << "cat_shape" << std::endl;
        PrintVec(cat_shape);
        temp_tensor.reshape(cat_shape);
    }
    std::cout << "input_shape" << std::endl;
    PrintVec(DiopiTensor(tensors[0]).shape());
    std::cout << "out_shape" << std::endl;
    PrintVec(DiopiTensor(out).shape());
    diopiCat(ctx, out, tensors, numTensors, dim);
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
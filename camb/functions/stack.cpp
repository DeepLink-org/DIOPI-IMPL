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
        auto temp_tensor = DiopiTensor(tensors[i]);                    // 从tensors得到元素并构造DiopiTensor temp_tensor
        std::vector<int64_t> temp_tensor_shape = temp_tensor.shape();  // 得到temp_tensor_shape
        std::vector<int64_t> cat_shape = temp_tensor_shape;
        cat_shape.insert(cat_shape.begin(), 1);  // cat_shape为temp_tensor_shape在指定维度插入1
        std::cout << "cat_shape" << std::endl;
        PrintVec(cat_shape);
        temp_tensor.reshape(cat_shape);  // 将temp_tensor reshape为cat_shape
        std::cout << "temp_tensor.shape()" << std::endl;
        PrintVec(temp_tensor.shape());  // 查看temp_tensor.shape(), 修改成功！！
        std::cout << "just check" << std::endl;
        PrintVec(DiopiTensor(diopiConstTensorHandle_t(temp_tensor)).shape()); // 查看反复转换的结果, 修改失败！！
    }
    std::cout << "input_shape" << std::endl;
    PrintVec(DiopiTensor(tensors[0]).shape());  // 查看tensor[0]的shape，修改不成功！！
    std::cout << "out_shape" << std::endl;
    PrintVec(DiopiTensor(out).shape());
    diopiCat(ctx, out, tensors, numTensors, dim);
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
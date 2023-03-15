#ifndef IMPL_CAMB_COMMON_COMMON_HPP_
#define IMPL_CAMB_COMMON_COMMON_HPP_

#include <set>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

template <typename DiopiTensorT>
DiopiTensorT dataTypeCast(diopiContextHandle_t& ctx, const DiopiTensorT& src, diopiDtype_t destDtype);

DiopiTensor<diopiTensorHandle_t> makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar);

template <typename DiopiTensorT>
void dataTypeCast(diopiContextHandle_t ctx, DiopiTensorT& dest, const DiopiTensorT& src);

diopiDtype_t choiceDtype(const std::set<diopiDtype_t>& opSupportedDtypes);

template <typename DiopiTensorT>
void autoCastTensorType(diopiContextHandle_t ctx, std::vector<DiopiTensorT*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_COMMON_COMMON_HPP_

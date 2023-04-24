#ifndef IMPL_CAMB_COMMON_CNNL_SCALAR_HPP_
#define IMPL_CAMB_COMMON_CNNL_SCALAR_HPP_

#include "common.hpp"

namespace impl {
namespace camb {

class CnnlScalar {
public:
    CnnlScalar() { data_mlu_ = nullptr; }
    template <typename T>
    explicit CnnlScalar(T data) {
        DIOPI_CHECK_ABORT(set(data) == diopiSuccess, "%s", "failed to set data");
    }
    template <typename T>
    diopiError_t set(T data) {
        if (data_mlu_ == nullptr) {
            // Allocate the memory with the maximum possible number of bytes, enabling it to repeatedly execute the 'set' function.
            DIOPI_CHECK(cnrtMalloc(&data_mlu_, 8) == cnrtSuccess, "%s", "failed to malloc memory.");
        }
        DIOPI_CHECK(cnrtMemcpy(data_mlu_, &data, sizeof(T), cnrtMemcpyHostToDev) == cnrtSuccess, "%s", "failed to malloc memory.");
        return diopiSuccess;
    }

    CnnlScalar(const CnnlScalar& other) = delete;
    CnnlScalar(CnnlScalar&& other) = delete;
    CnnlScalar& operator=(const CnnlScalar& other) = delete;
    ~CnnlScalar() {
        if (data_mlu_ != nullptr) {
            cnrtFree(data_mlu_);
        }
    }
    const void* data() const { return data_mlu_; }

private:
    void* data_mlu_;
};

}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_COMMON_CNNL_SCALAR_HPP_

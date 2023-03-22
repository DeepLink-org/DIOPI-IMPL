#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

DIOPI_API diopiError_t
diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train) {
    if (train) {
        cnnlHandle_t handle = cnnlHandlePool.get(ctx);

        const auto input_tensor = DiopiTensor(input);
        auto output_tensor = DiopiTensor(out);
        auto mask_tensor = DiopiTensor(mask);

        CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc mask_desc(mask_tensor, CNNL_LAYOUT_ARRAY);

        // create and set the rand_generator
        cnnlRandGenerator_t generator;
        // MTGP32 algorithm performs better on MLU300 series than MLU200 series

        DIOPI_CALLCNNL(cnnlRandCreateGenerator(&generator, CNNL_RAND_RNG_MTGP32));
        // set the period to the generator
        DIOPI_CALLCNNL(cnnlRandSetMTGP32Period(generator, CNNL_RAND_MTGP32_P11213));

        // create and set the state
        size_t size = 0;
        DIOPI_CALLCNNL(cnnlRandGetMTGP32StateSize(generator, &size));
        void* state = nullptr;
        state = requiresBuffer(ctx, size).data();

        DIOPI_CALLCNNL(cnnlRandMakeMTGP32State(handle, generator, state));
        DIOPI_CALLCNNL(cnnlFusedDropout_v2(
            handle, generator, input_desc.get(), input_tensor.data(), p, state, mask_desc.get(), mask_tensor.data(), output_desc.get(), output_tensor.data()));
        DIOPI_CALLCNNL(cnnlRandDestroyGenerator(generator));
    } else {
        diopiScalar_t scale;
        scale.stype = diopi_dtype_float64;
        scale.fval = 1;
        diopiMulScalar(ctx, out, input, &scale);
    }
    return diopiSuccess;
}
DIOPI_API diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train) {
    diopiDropout(ctx, input, mask, input, p, train);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
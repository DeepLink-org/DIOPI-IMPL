/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t bmm(diopiContextHandle_t ctx, DiopiTensor mat1, DiopiTensor mat2, DiopiTensor out) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlTensorDesc mat1_desc(mat1, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat2_desc(mat2, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> bmm_bcast_desc;
    cnnlDataType_t comp_type;
    if (out.dtype() == diopi_dtype_float32) {
        comp_type = CNNL_DTYPE_FLOAT;
    } else if (out.dtype() == diopi_dtype_float16) {
        comp_type = CNNL_DTYPE_HALF;
    } else {
        set_last_error_string("%s", "matmul on support float or half.");
        return diopiDtypeNotSupported;
    }

    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc.get(), CNNL_MATMUL_DESC_COMPUTE_TYPE, &(comp_type), sizeof(cnnlDataType_t)));

    int32_t is_transa = 0;
    int32_t is_transb = 0;
    int32_t allow_tf32_i32 = 0;
    int32_t use_beta = 0;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc.get(), CNNL_MATMUL_DESC_TRANSA, &(is_transa), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc.get(), CNNL_MATMUL_DESC_TRANSB, &(is_transb), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc.get(), CNNL_MATMUL_ALLOW_TF32, &(allow_tf32_i32), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc.get(), CNNL_MATMUL_USE_BETA, &(use_beta), sizeof(int32_t)));

    size_t workspace_size = 0;
    int requestedAlgoCount = 1;
    int returnAlgoCount = 0;
    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> heuristic_result;
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> algo;
    DIOPI_CALLCNNL(cnnlGetMatMulAlgoHeuristic(handle,
                                              bmm_bcast_desc.get(),
                                              mat1_desc.get(),
                                              mat2_desc.get(),
                                              out_desc.get(),
                                              out_desc.get(),
                                              nullptr,
                                              requestedAlgoCount,
                                              &heuristic_result.get(),
                                              &returnAlgoCount));
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristic_result.get(), algo.get(), &workspace_size));

    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    float alpha = 1.0;
    float beta = 0.0;

    DIOPI_CALLCNNL(cnnlBatchMatMulBCast_v2(handle,
                                           bmm_bcast_desc.get(),
                                           algo.get(),
                                           &alpha,
                                           mat1_desc.get(),
                                           mat1.data(),
                                           mat2_desc.get(),
                                           mat2.data(),
                                           &beta,
                                           out_desc.get(),
                                           out.data(),
                                           workspace,
                                           workspace_size));
    return diopiSuccess;
}
}  // namespace

extern "C" diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto mat2_tensor = DiopiTensor(mat2);
    auto output_tensor = DiopiTensor(out);

    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, mat2_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, output_tensor, diopi_dtype_float32));
    }
    DIOPI_CALL(bmm(ctx, input_tensor, mat2_tensor, output_tensor));
    auto src_out = DiopiTensor(out);
    DIOPI_CALL(dataTypeCast(ctx, src_out, output_tensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl

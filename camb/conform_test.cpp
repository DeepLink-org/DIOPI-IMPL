/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/
#include <cnmlrt.h>
#include <cnnl.h>
#include <diopi/diopirt.h>
#include <diopi_register.h>

#include <cstdio>
#include <mutex>

<<<<<<< HEAD

#define CALL_CNRT(Expr)   {                                                         \
    ::cnrtRet_t ret = Expr;                                                         \
    if (ret != ::CNRT_RET_SUCCESS) {                                                \
        printf("call a cambrt function (%s) failed. return code=%d", #Expr, ret);   \
    }}                                                                              \
=======
#include "error.hpp"

#define CALL_CNRT(Expr)                                                               \
    {                                                                                 \
        ::cnrtRet_t ret = Expr;                                                       \
        if (ret != ::CNRT_RET_SUCCESS) {                                              \
            printf("call a cambrt function (%s) failed. return code=%d", #Expr, ret); \
        }                                                                             \
    }
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb

extern "C" {
void* camb_malloc(uint64_t bytes) {
    void* ptr;
    CALL_CNRT(::cnrtMalloc(&ptr, bytes));
    return ptr;
}

<<<<<<< HEAD
void camb_free(void* ptr) {
    CALL_CNRT(cnrtFree(ptr));
}
=======
void camb_free(void* ptr) { CALL_CNRT(cnrtFree(ptr)); }
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb

int32_t camb_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    cnrtQueue_t phStream;
    CALL_CNRT(cnrtCreateQueue(&phStream));
    *stream_handle_ptr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

int32_t camb_destroy_stream(diopiStreamHandle_t stream_handle) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CNRT(cnrtDestroyQueue(phStream));
    return diopiSuccess;
}

int32_t camb_synchronize_stream(diopiStreamHandle_t stream_handle) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CNRT(cnrtSyncQueue(phStream));
    return diopiSuccess;
}

int32_t camb_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
<<<<<<< HEAD
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void *>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV));
=======
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV));
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb
    return diopiSuccess;
}

int32_t camb_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
<<<<<<< HEAD
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void *>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2HOST));
=======
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2HOST));
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb
    return diopiSuccess;
}

int32_t camb_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
<<<<<<< HEAD
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void *>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2DEV));
=======
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2DEV));
>>>>>>> 3445db0fa9705c1c2c09ee67d1120f71558f38cb
    return diopiSuccess;
}

int32_t initLibrary() {
    diopiRegisterDeviceMallocFunc(camb_malloc);
    diopiRegisterDevMemFreeFunc(camb_free);
    diopiRegisterStreamCreateFunc(camb_make_stream);
    diopiRegisterStreamDestroyFunc(camb_destroy_stream);
    diopiRegisterSynchronizeStreamFunc(camb_synchronize_stream);
    diopiRegisterMemcpyD2HAsyncFunc(camb_memcpy_d2h_async);
    diopiRegisterMemcpyD2DAsyncFunc(camb_memcpy_d2d_async);
    diopiRegisterMemcpyH2DAsyncFunc(camb_memcpy_h2d_async);
    diopiRegisterGetLastErrorFunc(impl::camb::camb_get_last_error_string);

    return diopiSuccess;
}

int32_t finalizeLibrary() { return diopiSuccess; }

}  // extern "C"

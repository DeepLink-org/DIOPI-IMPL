/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../error.hpp"

#include <diopi/functions.h>

namespace impl {
namespace camb {

char strLastError[8192] = {0};
char strLastErrorOther[4096] = {0};
std::mutex mtxLastError;

const char* camb_get_last_error_string() {
    // consider cnrt version cnrtGetLastErr or cnrtGetLaislhhstError
    ::cnrtRet_t err = ::cnrtGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "camb error: %s, more infos: %s", ::cnrtGetErrorStr(err), strLastErrorOther);
    return strLastError;
}

extern "C" DIOPI_RT_API const char* diopiGetLastErrorString() { return camb_get_last_error_string(); }




// diopiSuccess                                      = 0,
//     diopiErrorOccurred                                = 1,
//     diopiNotInited                                    = 2,
//     diopiNoRegisteredStreamCreateFunction             = 3,
//     diopiNoRegisteredStreamDestoryFunction            = 4,
//     diopiNoRegisteredStreamSyncFunction               = 5,
//     diopiNoRegisteredDeviceMemoryMallocFunction       = 6,
//     diopiNoRegisteredDeviceMemoryFreeFunction         = 7,
//     diopiNoRegisteredDevice2DdeviceMemoryCopyFunction = 8,
//     diopiNoRegisteredDevice2HostMemoryCopyFunction    = 9,
//     diopiNoRegisteredHost2DeviceMemoryCopyFunction    = 10,
//     diopiNoRegisteredGetLastErrorFunction             = 11,
//     diopi5DNotSupported                               = 12,
//     diopiDtypeNotSupported

const std::string getDiopiErrorStr(diopiError_t err) {
    switch (err) {
        case diopiErrorOccurred:
            return "diopiErrorOccurred";
        case diopiNotInited:
            return "diopiNotInited";
        case diopiNoRegisteredStreamCreateFunction:
            return "diopiNoRegisteredStreamCreateFunction";
        case diopiNoRegisteredStreamDestoryFunction:
            return "diopiNoRegisteredStreamDestoryFunction";
        case diopiNoRegisteredStreamSyncFunction:
            return "diopiNoRegisteredStreamSyncFunction";
        case diopiNoRegisteredDeviceMemoryMallocFunction:
            return "diopiNoRegisteredDeviceMemoryMallocFunction";
        case diopiNoRegisteredDeviceMemoryFreeFunction:
            return "diopiNoRegisteredDeviceMemoryFreeFunction";
        case diopiNoRegisteredDevice2DdeviceMemoryCopyFunction:
            return "diopiNoRegisteredDevice2DdeviceMemoryCopyFunction";
        case diopiNoRegisteredDevice2HostMemoryCopyFunction:
            return "diopiNoRegisteredDevice2HostMemoryCopyFunction";
        case diopiNoRegisteredHost2DeviceMemoryCopyFunction:
            return "diopiNoRegisteredHost2DeviceMemoryCopyFunction";
        case diopiNoRegisteredGetLastErrorFunction:
            return "diopiNoRegisteredGetLastErrorFunction";
        case diopi5DNotSupported:
            return "diopi5DNotSupported";
        case diopiDtypeNotSupported:
            return "diopiDtypeNotSupported";
    }
}

}  // namespace camb

}  // namespace impl

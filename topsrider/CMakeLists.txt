cmake_minimum_required(VERSION 3.4)
project(tops_impl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include_directories(
    /usr/include/dtu/
    /usr/include/dtu/topsdnn
    /usr/include/dtu/3_0/runtime/tops
    /usr/include/dtu/3_0/runtime
)

set(IMPL_SRC conform_test.cpp functions.cpp error.cpp)

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ${DIOPIRT})
target_link_libraries(${DEVICEIMPL} dtu_sdk)
target_link_libraries(${DEVICEIMPL} topsdnn)

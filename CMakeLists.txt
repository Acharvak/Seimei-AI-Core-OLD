cmake_minimum_required(VERSION 3.12)
project(Seimei LANGUAGES CXX)

# Settings should later be made more easily configurable
set(ENABLE_TESTS TRUE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# TODO: do not hardcode it
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")

# FIXME: get rid of this hack, find out how to do it the right way
if(CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_DEFINED_MACROS)
list(FIND CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_DEFINED_MACROS __cplusplus idx)
math(EXPR idx "${idx} + 1")
list(REMOVE_AT CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_DEFINED_MACROS ${idx})
list(INSERT CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_DEFINED_MACROS ${idx} 201703L)
set(CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_DEFINED_MACROS "${CMAKE_EXTRA_GENERATOR_CXX_SYSTEM_DEFINED_MACROS}" CACHE INTERNAL "" FORCE)
endif()

# FIXME
if(NOT BLAS_ROOT)
message(FATAL_ERROR "Please manually set -D BLAS_ROOT=/path/to/openblas")
endif(NOT BLAS_ROOT)

# Searching for third-party libraries
find_package(BLAS REQUIRED)    # Set BLAS_ROOT to override

if(ENABLE_TESTS)
    add_subdirectory(lib/googletest/googletest)
    include(GoogleTest)
endif(ENABLE_TESTS)

include_directories(SYSTEM lib/nlohmann-json/single_include)

include_directories(seimei/src)
include_directories(SYSTEM libPShowdown/include)

# FIXME: this should be done properly, but FindBLAS doesn't make it easy
include_directories("${BLAS_ROOT}/include")

# FIXME: this should be done properly
include_directories(/usr/include/hdf5/serial)

if(ENABLE_TESTS)
    enable_testing()
    add_executable(test_nnet-quick seimei/test/nnet-quick.cpp seimei/src/nnet/framework.cpp)
    gtest_discover_tests(test_nnet-quick)
    add_executable(test_nnet-iris seimei/test/nnet-iris.cpp seimei/src/nnet/framework.cpp)
    gtest_discover_tests(test_nnet-iris)
    target_link_libraries(test_nnet-quick gtest_main ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
    target_link_libraries(test_nnet-iris gtest_main ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif(ENABLE_TESTS)

add_executable(serialize-battle temp/serialize-battle.cpp)

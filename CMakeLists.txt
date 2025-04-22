cmake_minimum_required(VERSION 3.16...3.22)

find_program(CLANGPP "clang++")
set(CMAKE_CXX_COMPILER ${CLANGPP} CACHE STRING "C++ compiler" FORCE)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

project (
    batch_cc
    VERSION 0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(batch_cc
    scripts/batch_cc.cpp
    src/batch_cc.cu
)

target_include_directories(batch_cc
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(batch_cc
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)
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
find_package(Boost REQUIRED COMPONENTS graph regex)

### batch cc with motion bench maker problems
add_executable(batch_cc_mbm
    scripts/batch_cc_mbm.cpp
    src/batch_cc_copy.cu
)

target_include_directories(batch_cc_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(batch_cc_mbm
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

### batch cc with edges taken from graph.dot and scene taken from scene.txt
add_executable(batch_cc
    scripts/batch_cc.cpp
    src/batch_cc_bisect.cu
    # src/batch_cc_copy.cu
)

target_compile_options(batch_cc
    PRIVATE
    -g
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_include_directories(batch_cc
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(batch_cc
    PRIVATE
    Eigen3::Eigen
    Boost::boost
    Boost::graph
    Boost::regex
)
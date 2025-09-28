cmake_minimum_required(VERSION 3.16...3.22)

find_program(CLANGPP "clang++")
set(CMAKE_CXX_COMPILER ${CLANGPP} CACHE STRING "C++ compiler" FORCE)

include(CMakeFindDependencyMacro)

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

# Enable CUDA debugging
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")


# set(vamp_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vamp/install/share/cmake/vamp)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS graph regex)
# find_package(vamp REQUIRED)

FetchContent_Declare(vamp
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vamp
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/vamp-build
)
# Disable Python bindings for VAMP when used as a subproject
set(VAMP_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vamp)

### batch cc with motion bench maker problems
add_executable(batch_cc_mbm
    scripts/batch_cc_mbm.cpp
    src/batch_cc.cu
)

target_include_directories(batch_cc_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(batch_cc_mbm
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    vamp::vamp
)

### batch cc with edges taken from graph.dot and scene taken from scene.txt
add_executable(batch_cc
    scripts/batch_cc.cpp
    src/batch_cc.cu
)

target_compile_options(batch_cc
    PRIVATE
    -g
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
    vamp::vamp
)

### batch validate cc
add_executable(batch_cc_validate
    scripts/batch_cc_validate.cpp
    src/batch_cc.cu
)

target_compile_options(batch_cc_validate
    PRIVATE
    -mavx2 -mfma
)

target_include_directories(batch_cc_validate
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(batch_cc_validate
    PRIVATE
    Eigen3::Eigen
    Boost::boost
    Boost::graph
    Boost::regex
    vamp::vamp
)

# add_executable(batch_cc_validate1
#     scripts/batch_cc_validate1.cpp
#     src/batch_cc.cu
# )

# target_compile_options(batch_cc_validate1
#     PRIVATE
#     -mavx2 -mfma
# )

# target_include_directories(batch_cc_validate1
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )
# target_link_libraries(batch_cc_validate1
#     PRIVATE
#     Eigen3::Eigen
#     Boost::boost
#     Boost::graph
#     Boost::regex
#     vamp::vamp
# )
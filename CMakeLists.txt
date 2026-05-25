cmake_minimum_required(VERSION 3.20)
project(gpu-stress-tester LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Suppress pedantic warnings from nvcc-generated code (very common with CUDA + -Wpedantic)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-pedantic")

add_compile_options(-Wall -Wextra -Wpedantic)


add_library(gpu-stress-lib STATIC src/BenchmarkRunner.cpp src/CpuMatmulBenchmark.cpp src/GpuVectorAddBenchmark.cu )
target_include_directories(gpu-stress-lib PUBLIC include)

# Suppress noisy pedantic warnings from nvcc-generated code
# (common when using -Wpedantic with CUDA)
target_compile_options(gpu-stress-lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-pedantic>
)

add_executable(gpu-stress src/main.cpp)
target_link_libraries(gpu-stress PRIVATE gpu-stress-lib)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)

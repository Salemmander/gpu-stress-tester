cmake_minimum_required(VERSION 3.20)
project(gpu-stress-tester LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)


add_library(gpu-stress-lib STATIC src/BenchmarkRunner.cpp src/CpuMatmulBenchmark.cpp)
target_include_directories(gpu-stress-lib PUBLIC include)

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

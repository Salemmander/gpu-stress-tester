add_executable(test_benchmark test_benchmark.cpp)
target_link_libraries(test_benchmark PRIVATE  gpu-stress-lib gtest_main)
include(GoogleTest)
gtest_discover_tests(test_benchmark)

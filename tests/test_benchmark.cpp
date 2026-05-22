#include "Benchmark.h"
#include "BenchmarkRunner.h"
#include "CpuMatmulBenchmark.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

class FakeBenchmark : public Benchmark {
public:
  std::string name() const override;
  void setup() override;
  void run() override;
  void verify() override;
  void teardown() override;

  int setup_count = 0;
  int run_count = 0;
  int verify_count = 0;
  int teardown_count = 0;
  std::vector<std::string> order;
};

std::string FakeBenchmark::name() const { return "Fake"; };
void FakeBenchmark::setup() {
  setup_count++;
  order.push_back("setup");
}
void FakeBenchmark::run() {
  run_count++;
  order.push_back("run");
}
void FakeBenchmark::verify() {
  verify_count++;
  order.push_back("verify");
}
void FakeBenchmark::teardown() {
  teardown_count++;
  order.push_back("teardown");
}

TEST(BenchmarkRunner, CallsLifecycleCorrectly) {
  FakeBenchmark fake;
  BenchmarkRunner runner{4};

  auto result = runner.run(fake);

  std::vector<std::string> expected_order{"setup", "run", "verify", "teardown",
                                          "setup", "run", "verify", "teardown",
                                          "setup", "run", "verify", "teardown",
                                          "setup", "run", "verify", "teardown"};

  EXPECT_EQ(result.iterations, 4);
  EXPECT_EQ(fake.setup_count, 4);
  EXPECT_EQ(fake.run_count, 4);
  EXPECT_EQ(fake.verify_count, 4);
  EXPECT_EQ(fake.teardown_count, 4);
  EXPECT_EQ(expected_order, fake.order);
}

class CpuMatmulTestHelper {
public:
  static const std::vector<float> &getResult(const CpuMatmulBenchmark &bm) {
    return bm.c_;
  }
};

TEST(CpuMatmulBenchmark, CorrectResultFor1x1) {
  CpuMatmulBenchmark bm{1};
  ASSERT_EQ(bm.name(), "cpu-matmul");

  bm.setup();
  bm.run();
  EXPECT_NO_THROW(bm.verify());

  auto &result = CpuMatmulTestHelper::getResult(bm);
  EXPECT_EQ(std::vector<float>{1}, result);

  bm.teardown();
}

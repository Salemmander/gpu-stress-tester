#pragma once

#include <string>

class Benchmark {
public:
  virtual ~Benchmark() = default;
  virtual std::string name() const = 0;
  virtual void setup() = 0;
  virtual void run() = 0;
  virtual void verify() = 0;
  virtual void teardown() = 0;
};

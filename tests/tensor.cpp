#include <functional>
#include <iostream>
#include <stdexcept>

#include "dl/dl.h"

#include "test_macros.hpp"

int test_move_semantics() {
  dl::Tensor<int> a({2, 3});
  a.fill(5);
  int *original_ptr = a.data();

  dl::Tensor<int> b(std::move(a));

  // b must have stolen the data
  CHECK(b.size() == 6);
  CHECK(b.at({0,0}) == 5);

  // a must be empty/valid
  CHECK(a.size() == 0);
  CHECK(a.ndim() == 0);
  CHECK_THROWS_AS(a[0], std::out_of_range);
  return 0;
}

int test_bounds_checking() {
  dl::Tensor<int> t({2, 3}); // shape [2, 3], size 6

  // Out-of-bounds per dimension
  CHECK_THROWS_AS(t.at({2, 0}), std::out_of_range); // Row 2 doesn't exist
  CHECK_THROWS_AS(t.at({0, 3}), std::out_of_range); // Col 3 doesn't exist

  // Wrong number of dimensions
  CHECK_THROWS_AS(t.at({1}), std::invalid_argument);      // 1D index for 2D tensor
  CHECK_THROWS_AS(t.at({1, 2, 3}), std::invalid_argument); // 3D index

  // operator[] raw flat index bounds
  CHECK_THROWS_AS(t[6], std::out_of_range);
  CHECK_THROWS_AS(t[100], std::out_of_range);
  return 0;
}

int test_empty_tensor() {
  // An empty shape is valid (scalar 0-dim, or 0-sized vector)
  dl::Tensor<float> empty({});
  CHECK(empty.ndim() == 0);
  CHECK(empty.size() == 0);
  // Accessing it should throw, not crash
  CHECK_THROWS_AS(empty[0], std::out_of_range);

  // Tensor with a zero dimension
  dl::Tensor<int> zero_dim({5, 0, 3});
  CHECK(zero_dim.size() == 0);
  return 0;
}

int test_strides_and_flat_index() {
  dl::Tensor<int> t({2, 3, 4});

  // Expected strides: [12, 4, 1]
  std::vector<size_t> expected_strides = {12, 4, 1};
  CHECK(t.strides() == expected_strides);

  // Equivalence test: Set via multi-index, read via flat-index
  t.at({1, 2, 3}) = 42;
  size_t flat_idx = 1*12 + 2*4 + 3*1; // 12 + 8 + 3 = 23
  CHECK(t[flat_idx] == 42);

  // Set via flat, read via multi
  t[10] = 99;
  // 10 = 0*12 + 2*4 + 2*1 -> index [0, 2, 2]
  CHECK(t.at({0, 2, 2}) == 99);

  return 0;
}

int test_reassignment_and_resize() {
  dl::Tensor<int> t({2, 3});
  size_t old_size = t.size();

  // Reassign to a larger tensor
  t = dl::Tensor<int>({10, 10});
  // Old shared_ptr ref count drops. If no other refs, memory frees.
  CHECK(t.size() == 100);
  CHECK(t.ndim() == 2);
  // Ensure old data is zero-initialized (constructor fills with 0)
  CHECK(t.at({9,9}) == 0);

  return 0;
}

int test_large_allocation_no_crash() {
  // Allocate ~100MB. This shouldn't throw std::bad_alloc.
  dl::Tensor<double> large({1000, 1000, 10}); // 10 million elements
  CHECK(large.size() == 10'000'000);
  large.fill(3.14);
  CHECK(large.at({999, 999, 9}) == 3.14); // Bounds check on edge indices

  return 0;
}

int test_matmul() {
  dl::Tensor<int> A({2, 3});
  A.at({0, 0}) = 1; A.at({0, 1}) = 2; A.at({0, 2}) = 3;
  A.at({1, 0}) = 4; A.at({1, 1}) = 5; A.at({1, 2}) = 6;

  dl::Tensor<int> B({3, 2});
  B.at({0, 0}) = 7;  B.at({0, 1}) = 8;
  B.at({1, 0}) = 9;  B.at({1, 1}) = 10;
  B.at({2, 0}) = 11; B.at({2, 1}) = 12;

  dl::Tensor<int> C = A.matmul(B);

  CHECK(C.shape()[0] == 2);
  CHECK(C.shape()[1] == 2);

  // Verify values
  CHECK(C.at({0, 0}) == 58);
  CHECK(C.at({0, 1}) == 64);
  CHECK(C.at({1, 0}) == 139);
  CHECK(C.at({1, 1}) == 154);

  CHECK_THROWS_AS(A.matmul(A), std::invalid_argument);

  return 0;
}

int test_broadcasting() {
  dl::Tensor<int> a({2, 3});
  dl::Tensor<int> b({2, 1});
  dl::Tensor<int> c = a + b;
  return 0;
}

int main() {
  struct TestCase {
    std::string name;
    std::function<int()> func;
  };

  std::vector<TestCase> tests;
  tests.push_back({"move_semantics", test_move_semantics});
  tests.push_back({"bounds_checking", test_bounds_checking});
  tests.push_back({"empty_tensor", test_empty_tensor});
  tests.push_back({"strides_and_flat_index", test_strides_and_flat_index});
  tests.push_back({"reassignment_and_resize", test_reassignment_and_resize});
  tests.push_back({"large_allocation_no_crash", test_large_allocation_no_crash});
  tests.push_back({"matrix_multiplication", test_matmul});
  tests.push_back({"broadcasting", test_broadcasting});

  int passed = 0;
  int failed = 0;

  for (const auto &test: tests) {
    std::cout << "[ RUN ] " << test.name << ":  " << std::flush;

    try {
      if (test.func() == 0) {
        std::cout << "PASS\n";
        ++passed;
      } else {
        std::cout << "FAIL\n";
        ++failed;
      }
    } catch (const std::exception &e) {
      std::cout << "FAIL (Exception: " << e.what() << ")\n";
      ++failed;
    } catch (...) {
      std::cout << "FAIL (Unknown/Non-standard exception)\n";
      ++failed;
    }
  }

  std::cout << "Results: " << passed << " passed, " << failed << " failed\n";

  return (failed == 0 ? 0 : 1);
}

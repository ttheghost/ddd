//
// Created by info on 06/07/2026.
//

#pragma once

#include <iostream>
#include <stdexcept>

#define CHECK(expr)                                                                           \
  do {                                                                                        \
    if (!(expr)) {                                                                            \
      std::cerr << "Test failed: " << #expr << " is false\n";                                 \
      return 1;                                                                               \
    }                                                                                         \
  } while (false)

#define CHECK_THROWS_AS(expr, exception_type)                                                 \
  do {                                                                                        \
    bool caught_expected = false;                                                             \
    try {                                                                                     \
      expr;                                                                                   \
      std::cerr << "Test failed: " << #expr << " did not throw any exception.\n";             \
    } catch (const exception_type&) {                                                         \
      caught_expected = true;                                                                 \
    } catch (const std::exception& e) {                                                       \
      std::cerr << "Test failed: " << #expr << " threw an unexpected exception: "             \
                << e.what() << "\n";                                                          \
    } catch (...) {                                                                           \
      std::cerr << "Test failed: " << #expr << " threw an unknown exception type.\n";         \
    }                                                                                         \
    if (!caught_expected) {                                                                   \
      return 1;                                                                               \
    }                                                                                         \
  } while (false)

// CREATE KEYSPACE bookstore WITH replacation = {...};

// CREATE TABLE
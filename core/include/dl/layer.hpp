#pragma once
#include "tensor.hpp"

namespace dl {
  template<typename T>
    requires std::is_arithmetic_v<T>
  class Layer {
  public:
    virtual ~Layer() = default;

    virtual Tensor<T> forward(const Tensor<T> &) = 0;

    virtual Tensor<T> backward(const Tensor<T> &) = 0;

    /** Toggle training vs inference mode.  Default: no-op (stateless layers). */
    virtual void set_training(bool /*training*/) {
    }
  };
}

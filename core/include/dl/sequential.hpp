#pragma once
#include "layer.hpp"

#include <memory>
#include <vector>

namespace dl {
  template<typename T>
    requires std::is_arithmetic_v<T>
  class Sequential {
    std::vector<std::unique_ptr<Layer<T> > > layers_;

  public:
    void add(std::unique_ptr<Layer<T> > layer) {
      layers_.push_back(std::move(layer));
    }

    Tensor<T> forward(const Tensor<T> &input) {
      Tensor<T> x = input;
      for (auto &l: layers_) x = l->forward(x);
      return x;
    }

    Tensor<T> backward(const Tensor<T> &grad) {
      Tensor<T> g = grad;
      for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        g = (*it)->backward(g);
      return g;
    }

    /** Propagate training/inference mode to every layer. */
    void set_training(bool training) {
      for (auto &l: layers_) l->set_training(training);
    }
  };
}

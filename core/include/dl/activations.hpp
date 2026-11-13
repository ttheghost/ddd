/**
 * @file activations.hpp
 * @brief Stateless and stateful activation layers:
 *        ReLU, Sigmoid, Tanh, Softmax.
 *
 * All activations save the minimum state needed for backward.
 *
 * Softmax backward note
 * ─────────────────────
 * This Softmax backward is designed for the CE+Softmax combined gradient
 * shortcut:  grad_loss = pred - y  is the gradient of the cross-entropy loss
 * w.r.t. the pre-softmax logits z.  Because that combined gradient is already
 * ∂L/∂z (not ∂L/∂softmax(z)), the Softmax backward simply passes it through
 * to the Dense layer below.
 *
 * Derivation why this is correct:
 *   CE loss:        L = -Σ y_i log(s_i)
 *   ∂L/∂s_i       = -y_i / s_i
 *   Softmax Jacobian: ∂s_i/∂z_j = s_i(δ_ij - s_j)
 *   ∂L/∂z_i       = Σ_j (∂L/∂s_j)(∂s_j/∂z_i)
 *                 = s_i(∂L/∂s_i - Σ_j s_j ∂L/∂s_j)
 *                 = s_i(-y_i/s_i - Σ_j s_j(-y_j/s_j))
 *                 = -y_i + s_i * Σ_j y_j
 *                 = s_i - y_i     (since Σy_j = 1 for one-hot)
 *   So if we pass (pred - y) as grad_output, it is already ∂L/∂z.
 *   → return it unchanged.
 */

#pragma once
#include "layer.hpp"
#include "tensor.hpp"

namespace dl {
  // ReLU
  // forward : y  = max(0, x)
  // backward: g_x = g_y * (x > 0)

  template<typename T>
    requires std::is_arithmetic_v<T>
  class ReLU final : public Layer<T> {
    Tensor<T> saved_input_{{1}};

  public:
    Tensor<T> forward(const Tensor<T> &input) override {
      saved_input_ = input;
      Tensor<T> result(input.shape());
      for (size_t i = 0; i < input.size(); ++i)
        result.data()[i] = std::max(T{0}, input[i]);
      return result;
    }

    Tensor<T> backward(const Tensor<T> &grad) override {
      Tensor<T> result(grad.shape());
      for (size_t i = 0; i < grad.size(); ++i)
        result.data()[i] = (saved_input_[i] > T{0}) ? grad[i] : T{0};
      return result;
    }
  };

  // Sigmoid
  // forward : s  = 1 / (1 + exp(-x))
  // backward: g_x = g_y * s*(1-s)

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Sigmoid final : public Layer<T> {
    Tensor<T> saved_output_{{1}};

  public:
    Tensor<T> forward(const Tensor<T> &input) override {
      saved_output_ = T{1} / (T{1} + (-input).exp());
      return saved_output_;
    }

    Tensor<T> backward(const Tensor<T> &grad) override {
      return grad * saved_output_ * (T{1} - saved_output_);
    }
  };

  // Tanh
  // forward : t  = tanh(x)
  // backward: g_x = g_y * (1 - t^2)

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Tanh final : public Layer<T> {
    Tensor<T> saved_output_{{1}};

  public:
    Tensor<T> forward(const Tensor<T> &input) override {
      saved_output_ = input.tanh();
      return saved_output_;
    }

    Tensor<T> backward(const Tensor<T> &grad) override {
      // d/dx tanh(x) = 1 - tanh²(x)
      return grad * (T{1} - saved_output_ * saved_output_);
    }
  };

  // Softmax
  // forward : s_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))  [numerically stable]
  // backward: pass-through (designed for CE+Softmax combined gradient use)
  //           see file-level docstring for full derivation.

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Softmax final : public Layer<T> {
    Tensor<T> saved_output_{{1}};

  public:
    Tensor<T> forward(const Tensor<T> &input) override {
      Tensor<T> shifted = input - input.max();
      Tensor<T> exps = shifted.exp();
      saved_output_ = exps / exps.sum();
      return saved_output_;
    }

    /**
     * Pass-through backward for CE+Softmax combined use.
     * The caller must pass  (pred - y_onehot)  as grad_output,
     * which IS the combined ∂L/∂z already computed analytically.
     */
    Tensor<T> backward(const Tensor<T> &grad_output) override {
      return grad_output;
    }
  };
} // namespace dl

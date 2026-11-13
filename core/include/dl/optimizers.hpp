/**
 * @file optimizers.hpp
 * @brief SGD, SGDMomentum, and Adam optimizers.
 *
 * Usage:
 *   dl::Adam<float> adam(0.001f);
 *   // pass to a Dense layer — it clones the prototype for W and b
 *   model.add(std::make_unique<dl::Dense<float>>(784, 256, adam));
 *
 * All optimizers lazily allocate their moment tensors on the first call to
 * step(), so they don't need to know the parameter shape at construction.
 * clone() returns a fresh instance with the same hyperparameters and zeroed
 * state — used by Dense to create independent optimizer copies for W and b.
 */

#pragma once
#include "tensor.hpp"
#include <cmath>
#include <limits>
#include <memory>

namespace dl {
  // Abstract base

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Optimizer {
  public:
    virtual ~Optimizer() = default;

    /** In-place parameter update: param -= f(grad). */
    virtual void step(Tensor<T> &param, const Tensor<T> &grad) = 0;

    /** Return a fresh clone with same hyper-params and zeroed state. */
    [[nodiscard]] virtual std::unique_ptr<Optimizer<T> > clone() const = 0;
  };

  // Vanilla SGD

  template<typename T>
    requires std::is_arithmetic_v<T>
  class SGD final : public Optimizer<T> {
    T lr_;

  public:
    explicit SGD(T lr = T{0.01}) : lr_(lr) {
    }

    void step(Tensor<T> &param, const Tensor<T> &grad) override {
      param -= lr_ * grad;
    }

    [[nodiscard]] std::unique_ptr<Optimizer<T> > clone() const override {
      return std::make_unique<SGD<T> >(lr_);
    }
  };

  // SGD with Momentum
  //
  // v  ← beta * v  +  grad          (PyTorch-style heavy-ball)
  // p  ← p  -  lr * v

  template<typename T>
    requires std::is_arithmetic_v<T>
  class SGDMomentum final : public Optimizer<T> {
    T lr_, beta_;
    Tensor<T> velocity_{{1}};
    bool initialized_ = false;

  public:
    SGDMomentum(T lr = T{0.01}, T beta = T{0.9}) : lr_(lr), beta_(beta) {
    }

    void step(Tensor<T> &param, const Tensor<T> &grad) override {
      if (!initialized_) {
        velocity_ = Tensor<T>(param.shape());
        velocity_.fill(T{0});
        initialized_ = true;
      }
      velocity_ = beta_ * velocity_ + grad;
      param -= lr_ * velocity_;
    }

    [[nodiscard]] std::unique_ptr<Optimizer<T> > clone() const override {
      return std::make_unique<SGDMomentum<T> >(lr_, beta_);
    }
  };

  // Adam
  //
  // m  ← beta1*m + (1-beta1)*g
  // v  ← beta2*v + (1-beta2)*g²
  // m̂  = m / (1 - beta1^t)
  // v̂  = v / (1 - beta2^t)
  // p  ← p - lr * m̂ / (√v̂ + ε)

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Adam final : public Optimizer<T> {
    T lr_, beta1_, beta2_, eps_;
    Tensor<T> m_{{1}};
    Tensor<T> v_{{1}};
    size_t t_ = 0;
    bool initialized_ = false;

  public:
    explicit Adam(T lr = T{0.001},
                  T beta1 = T{0.9},
                  T beta2 = T{0.999},
                  T eps = T{1e-8})
      : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    }

    void step(Tensor<T> &param, const Tensor<T> &grad) override {
      if (!initialized_) {
        m_ = Tensor<T>(param.shape());
        m_.fill(T{0});
        v_ = Tensor<T>(param.shape());
        v_.fill(T{0});
        initialized_ = true;
      }
      ++t_;

      // Update biased first/second moment estimates
      m_ = beta1_ * m_ + (T{1} - beta1_) * grad;
      v_ = beta2_ * v_ + (T{1} - beta2_) * (grad * grad);

      // Bias corrections
      const T bc1 = T{1} - std::pow(beta1_, static_cast<T>(t_));
      const T bc2 = T{1} - std::pow(beta2_, static_cast<T>(t_));
      auto m_hat = m_ / bc1;
      auto v_hat = v_ / bc2;

      // Clamp v_hat to avoid sqrt of near-zero negatives (numerical safety)
      auto v_safe = v_hat.clamp(T{0}, std::numeric_limits<T>::max());
      param -= lr_ * (m_hat / (v_safe.sqrt() + eps_));
    }

    [[nodiscard]] std::unique_ptr<Optimizer<T> > clone() const override {
      return std::make_unique<Adam<T> >(lr_, beta1_, beta2_, eps_);
    }
  };
} // namespace dl

/**
 * @file layers.hpp
 * @brief Parameterised and structural layers:
 *        Dense, Dropout, Reshape, Flatten, Conv2D, MaxPool2D, Embedding, RNNCell, LSTMCell, GRUCell, PositionalEncoding, SingleHeadAttention.
 */

#pragma once
#include "layer.hpp"
#include "optimizers.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace dl {
  // Dense (Fully-connected)
  //
  // forward : y = W x + b         W:(out,in)  x:(in,1)  b:(out,1)
  // backward: ∂L/∂x = W^T · g    (in,1)
  //           ΔW    = outer(g, x) (out,in)
  //           Δb    = g           (out,1)

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Dense final : public Layer<T> {
    Tensor<T> weights_; // (out, in)
    Tensor<T> bias_; // (out, 1)
    Tensor<T> saved_input_; // (in,  1)  - set by forward, read by backward

    std::unique_ptr<Optimizer<T> > opt_w_;
    std::unique_ptr<Optimizer<T> > opt_b_;

  public:
    /**
     * @param input_dim   Number of input features (fan_in)
     * @param output_dim  Number of output neurons  (fan_out)
     * @param opt_proto   Optimizer prototype — cloned for W and b separately
     * @param he_init     true → He init (ReLU); false → Xavier init (Sigmoid/Tanh)
     */
    Dense(size_t input_dim,
          size_t output_dim,
          const Optimizer<T> &opt_proto,
          const bool he_init = true)
      : weights_({output_dim, input_dim}),
        bias_({output_dim, 1}),
        saved_input_({input_dim, 1}),
        opt_w_(opt_proto.clone()),
        opt_b_(opt_proto.clone()) {
      bias_.fill(T{0});

      std::mt19937 rng(std::random_device{}());

      if (he_init) {
        // He / Kaiming:  W ~ N(0, sqrt(2/fan_in))
        const T std_dev = std::sqrt(T{2} / static_cast<T>(input_dim));
        std::normal_distribution<T> dist(T{0}, std_dev);
        for (size_t i = 0; i < weights_.size(); ++i)
          weights_.data()[i] = dist(rng);
      } else {
        // Xavier / Glorot:  W ~ U(-limit, +limit)
        const T limit = std::sqrt(T{6} / static_cast<T>(input_dim + output_dim));
        std::uniform_real_distribution<T> dist(-limit, limit);
        for (size_t i = 0; i < weights_.size(); ++i)
          weights_.data()[i] = dist(rng);
      }
    }

    Tensor<T> forward(const Tensor<T> &input) override {
      saved_input_ = input; // shallow copy (safe: read-only in bwd)
      return weights_.matmul(input) + bias_; // (out,1)
    }

    Tensor<T> backward(const Tensor<T> &grad_output) override {
      // Gradient w.r.t. input -> pass to the layer below
      auto grad_input = weights_.transpose().matmul(grad_output); // (in,1)

      // Gradient w.r.t. weights and bias -> update via optimizers
      auto grad_w = Tensor<T>::outer(grad_output, saved_input_); // (out,in)
      opt_w_->step(weights_, grad_w);
      opt_b_->step(bias_, grad_output);

      return grad_input;
    }

    /** Direct access to weight / bias storage for custom initialization. */
    T *weights_data() { return weights_.data(); }
    const T *weights_data() const { return weights_.data(); }
    [[nodiscard]] size_t weights_size() const { return weights_.size(); }

    T *bias_data() { return bias_.data(); }
    const T *bias_data() const { return bias_.data(); }
    [[nodiscard]] size_t bias_size() const { return bias_.size(); }
  };

  // Dropout
  //
  // forward (train): mask each element with Bernoulli(1-p), scale by 1/(1-p)
  // forward (eval) : identity
  // backward       : multiply by saved mask (or identity in eval)

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Dropout final : public Layer<T> {
    T drop_prob_; // probability of zeroing an element
    Tensor<T> mask_{{1}};
    bool training_ = true;
    std::mt19937 rng_;

  public:
    /**
     * @param drop_prob  Probability of dropping a unit (0 = keep all, 0.5 = drop half)
     * @param seed       RNG seed (default: hardware random)
     */
    explicit Dropout(T drop_prob, unsigned seed = std::random_device{}())
      : drop_prob_(drop_prob), rng_(seed) {
    }

    void set_training(bool training) override { training_ = training; }

    Tensor<T> forward(const Tensor<T> &input) override {
      if (!training_) return input;

      mask_ = Tensor<T>(input.shape());
      const T scale = T{1} / (T{1} - drop_prob_);
      std::bernoulli_distribution dist(static_cast<double>(T{1} - drop_prob_));

      for (size_t i = 0; i < mask_.size(); ++i)
        mask_.data()[i] = dist(rng_) ? scale : T{0};

      return input * mask_;
    }

    Tensor<T> backward(const Tensor<T> &grad) override {
      if (!training_) return grad;
      return grad * mask_;
    }
  };

  // Reshape

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Reshape final : public Layer<T> {
    std::vector<size_t> target_shape_;
    std::vector<size_t> saved_input_shape_;

  public:
    explicit Reshape(std::vector<size_t> target_shape)
      : target_shape_(std::move(target_shape)) {
    }

    Tensor<T> forward(const Tensor<T> &input) override {
      saved_input_shape_ = input.shape();
      return input.reshape(target_shape_);
    }

    Tensor<T> backward(const Tensor<T> &grad_output) override {
      return grad_output.reshape(saved_input_shape_);
    }
  };

  // Flatten

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Flatten final : public Layer<T> {
    std::vector<size_t> saved_input_shape_;

  public:
    Tensor<T> forward(const Tensor<T> &input) override {
      saved_input_shape_ = input.shape();
      return input.reshape({input.size(), 1});
    }

    Tensor<T> backward(const Tensor<T> &grad_output) override {
      return grad_output.reshape(saved_input_shape_);
    }
  };

  // Conv2D

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Conv2D final : public Layer<T> {
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;

    Tensor<T> weights_; // (out_c, in_c, k, k)
    Tensor<T> bias_; // (out_c, 1, 1)

    Tensor<T> saved_input_{{1}};
    std::vector<size_t> saved_input_shape_;

    std::unique_ptr<Optimizer<T> > opt_w_;
    std::unique_ptr<Optimizer<T> > opt_b_;

  public:
    Conv2D(size_t in_channels,
           size_t out_channels,
           size_t kernel_size,
           size_t stride,
           size_t padding,
           const Optimizer<T> &opt_proto,
           bool he_init = true)
      : in_channels_(in_channels),
        out_channels_(out_channels),
        kernel_size_(kernel_size),
        stride_(stride),
        padding_(padding),
        weights_({out_channels, in_channels, kernel_size, kernel_size}),
        bias_({out_channels, 1, 1}),
        opt_w_(opt_proto.clone()),
        opt_b_(opt_proto.clone()) {
      bias_.fill(T{0});
      std::mt19937 rng(std::random_device{}());
      const size_t fan_in = in_channels * kernel_size * kernel_size;
      const T std_dev = he_init
                          ? std::sqrt(T{2} / static_cast<T>(fan_in))
                          : std::sqrt(T{1} / static_cast<T>(fan_in));
      std::normal_distribution<T> dist(T{0}, std_dev);
      for (size_t i = 0; i < weights_.size(); ++i)
        weights_.data()[i] = dist(rng);
    }

    Tensor<T> forward(const Tensor<T> &input) override {
      saved_input_shape_ = input.shape();

      // Ensure 3D input shape (in_channels, in_h, in_w)
      Tensor<T> in_3d = input;
      if (input.ndim() == 2 && input.shape()[1] == 1) {
        // e.g. (C*H*W, 1) -> infer square H, W for in_channels_
        const size_t spatial = input.shape()[0] / in_channels_;
        const size_t side = static_cast<size_t>(std::sqrt(spatial));
        in_3d = input.reshape({in_channels_, side, side});
      } else if (input.ndim() != 3) {
        throw std::invalid_argument("Conv2D expected 3D tensor (C, H, W)");
      }

      saved_input_ = in_3d;

      const size_t in_h = in_3d.shape()[1];
      const size_t in_w = in_3d.shape()[2];

      const size_t out_h = (in_h - kernel_size_ + 2 * padding_) / stride_ + 1;
      const size_t out_w = (in_w - kernel_size_ + 2 * padding_) / stride_ + 1;

      Tensor<T> output({out_channels_, out_h, out_w});

      const T *x_ptr = in_3d.data();
      const T *w_ptr = weights_.data();
      const T *b_ptr = bias_.data();
      T *y_ptr = output.data();

      for (size_t o = 0; o < out_channels_; ++o) {
        const T b_val = b_ptr[o];
        for (size_t r = 0; r < out_h; ++r) {
          for (size_t c = 0; c < out_w; ++c) {
            T sum = b_val;
            for (size_t i = 0; i < in_channels_; ++i) {
              for (size_t m = 0; m < kernel_size_; ++m) {
                const int pr = static_cast<int>(r * stride_ + m) - static_cast<int>(padding_);
                if (pr < 0 || pr >= static_cast<int>(in_h)) continue;

                for (size_t n = 0; n < kernel_size_; ++n) {
                  const int pc = static_cast<int>(c * stride_ + n) - static_cast<int>(padding_);
                  if (pc < 0 || pc >= static_cast<int>(in_w)) continue;

                  const size_t x_idx = i * (in_h * in_w) + static_cast<size_t>(pr) * in_w + static_cast<size_t>(pc);
                  const size_t w_idx = o * (in_channels_ * kernel_size_ * kernel_size_)
                                       + i * (kernel_size_ * kernel_size_)
                                       + m * kernel_size_ + n;

                  sum += x_ptr[x_idx] * w_ptr[w_idx];
                }
              }
            }
            y_ptr[o * (out_h * out_w) + r * out_w + c] = sum;
          }
        }
      }

      return output;
    }

    Tensor<T> backward(const Tensor<T> &grad_output) override {
      const size_t in_h = saved_input_.shape()[1];
      const size_t in_w = saved_input_.shape()[2];

      const size_t out_h = grad_output.shape()[1];
      const size_t out_w = grad_output.shape()[2];

      Tensor<T> grad_w({out_channels_, in_channels_, kernel_size_, kernel_size_});
      Tensor<T> grad_b({out_channels_, 1, 1});
      Tensor<T> grad_x({in_channels_, in_h, in_w});

      grad_w.fill(T{0});
      grad_b.fill(T{0});
      grad_x.fill(T{0});

      const T *x_ptr = saved_input_.data();
      const T *w_ptr = weights_.data();
      const T *gy_ptr = grad_output.data();

      T *gw_ptr = grad_w.data();
      T *gb_ptr = grad_b.data();
      T *gx_ptr = grad_x.data();

      for (size_t o = 0; o < out_channels_; ++o) {
        for (size_t r = 0; r < out_h; ++r) {
          for (size_t c = 0; c < out_w; ++c) {
            const T go = gy_ptr[o * (out_h * out_w) + r * out_w + c];
            gb_ptr[o] += go;

            for (size_t i = 0; i < in_channels_; ++i) {
              for (size_t m = 0; m < kernel_size_; ++m) {
                const int pr = static_cast<int>(r * stride_ + m) - static_cast<int>(padding_);
                if (pr < 0 || pr >= static_cast<int>(in_h)) continue;

                for (size_t n = 0; n < kernel_size_; ++n) {
                  const int pc = static_cast<int>(c * stride_ + n) - static_cast<int>(padding_);
                  if (pc < 0 || pc >= static_cast<int>(in_w)) continue;

                  const size_t x_idx = i * (in_h * in_w) + static_cast<size_t>(pr) * in_w + static_cast<size_t>(pc);
                  const size_t w_idx = o * (in_channels_ * kernel_size_ * kernel_size_)
                                       + i * (kernel_size_ * kernel_size_)
                                       + m * kernel_size_ + n;

                  gw_ptr[w_idx] += go * x_ptr[x_idx];
                  gx_ptr[x_idx] += go * w_ptr[w_idx];
                }
              }
            }
          }
        }
      }

      opt_w_->step(weights_, grad_w);
      opt_b_->step(bias_, grad_b);

      if (saved_input_shape_.size() == 2) {
        return grad_x.reshape(saved_input_shape_);
      }
      return grad_x;
    }
  };

  // MaxPool2D

  template<typename T>
    requires std::is_arithmetic_v<T>
  class MaxPool2D final : public Layer<T> {
    size_t pool_size_;
    size_t stride_;

    std::vector<size_t> saved_input_shape_;
    std::vector<size_t> max_indices_; // Stores linear input index for max element

  public:
    explicit MaxPool2D(size_t pool_size = 2, size_t stride = 2)
      : pool_size_(pool_size), stride_(stride) {
    }

    Tensor<T> forward(const Tensor<T> &input) override {
      saved_input_shape_ = input.shape();

      if (input.ndim() != 3) {
        throw std::invalid_argument("MaxPool2D expected 3D tensor (C, H, W)");
      }

      const size_t channels = input.shape()[0];
      const size_t in_h = input.shape()[1];
      const size_t in_w = input.shape()[2];

      const size_t out_h = (in_h - pool_size_) / stride_ + 1;
      const size_t out_w = (in_w - pool_size_) / stride_ + 1;

      Tensor<T> output({channels, out_h, out_w});
      max_indices_.resize(channels * out_h * out_w);

      const T *x_ptr = input.data();
      T *y_ptr = output.data();

      for (size_t c = 0; c < channels; ++c) {
        for (size_t r = 0; r < out_h; ++r) {
          for (size_t col = 0; col < out_w; ++col) {
            T max_val = std::numeric_limits<T>::lowest();
            size_t max_idx = 0;

            for (size_t m = 0; m < pool_size_; ++m) {
              for (size_t n = 0; n < pool_size_; ++n) {
                const size_t pr = r * stride_ + m;
                const size_t pc = col * stride_ + n;
                const size_t in_idx = c * (in_h * in_w) + pr * in_w + pc;
                const T val = x_ptr[in_idx];

                if (val > max_val) {
                  max_val = val;
                  max_idx = in_idx;
                }
              }
            }

            const size_t out_idx = c * (out_h * out_w) + r * out_w + col;
            y_ptr[out_idx] = max_val;
            max_indices_[out_idx] = max_idx;
          }
        }
      }

      return output;
    }

    Tensor<T> backward(const Tensor<T> &grad_output) override {
      Tensor<T> grad_input(saved_input_shape_);
      grad_input.fill(T{0});

      T *gx_ptr = grad_input.data();
      const T *gy_ptr = grad_output.data();

      for (size_t i = 0; i < max_indices_.size(); ++i) {
        gx_ptr[max_indices_[i]] += gy_ptr[i];
      }

      return grad_input;
    }
  };

  // Embedding

  template<typename T>
    requires std::is_arithmetic_v<T>
  class Embedding final {
    size_t vocab_size_;
    size_t embed_dim_;
    Tensor<T> weights_; // (vocab_size, embed_dim)
    std::unique_ptr<Optimizer<T> > opt_;
    size_t saved_token_id_ = 0;

  public:
    Embedding(size_t vocab_size, size_t embed_dim, const Optimizer<T> &opt_proto)
      : vocab_size_(vocab_size),
        embed_dim_(embed_dim),
        weights_({vocab_size, embed_dim}),
        opt_(opt_proto.clone()) {
      std::mt19937 rng(std::random_device{}());
      const T std_dev = std::sqrt(T{1} / static_cast<T>(embed_dim));
      std::normal_distribution<T> dist(T{0}, std_dev);
      for (size_t i = 0; i < weights_.size(); ++i)
        weights_.data()[i] = dist(rng);
    }

    Tensor<T> forward(size_t token_id) {
      if (token_id >= vocab_size_) {
        throw std::out_of_range("Embedding token_id out of range");
      }
      saved_token_id_ = token_id;

      Tensor<T> embed({embed_dim_, 1});
      const T *w_ptr = weights_.data() + token_id * embed_dim_;
      T *out_ptr = embed.data();
      std::copy_n(w_ptr, embed_dim_, out_ptr);
      return embed;
    }

    void backward(const Tensor<T> &grad_output) {
      Tensor<T> grad_w({vocab_size_, embed_dim_});
      grad_w.fill(T{0});

      T *gw_ptr = grad_w.data() + saved_token_id_ * embed_dim_;
      const T *go_ptr = grad_output.data();
      std::copy_n(go_ptr, embed_dim_, gw_ptr);

      opt_->step(weights_, grad_w);
    }

    [[nodiscard]] size_t vocab_size() const { return vocab_size_; }
    [[nodiscard]] size_t embed_dim() const { return embed_dim_; }
  };

  // RNNCell

  template<typename T>
    requires std::is_arithmetic_v<T>
  class RNNCell final {
    size_t in_dim_;
    size_t hidden_dim_;

    Tensor<T> w_ih_; // (hidden_dim, in_dim)
    Tensor<T> w_hh_; // (hidden_dim, hidden_dim)
    Tensor<T> bias_; // (hidden_dim, 1)

    Tensor<T> saved_x_{{1}};
    Tensor<T> saved_h_prev_{{1}};
    Tensor<T> saved_h_{{1}};

    std::unique_ptr<Optimizer<T> > opt_w_ih_;
    std::unique_ptr<Optimizer<T> > opt_w_hh_;
    std::unique_ptr<Optimizer<T> > opt_b_;

  public:
    RNNCell(size_t in_dim, size_t hidden_dim, const Optimizer<T> &opt_proto)
      : in_dim_(in_dim),
        hidden_dim_(hidden_dim),
        w_ih_({hidden_dim, in_dim}),
        w_hh_({hidden_dim, hidden_dim}),
        bias_({hidden_dim, 1}),
        opt_w_ih_(opt_proto.clone()),
        opt_w_hh_(opt_proto.clone()),
        opt_b_(opt_proto.clone()) {
      bias_.fill(T{0});
      std::mt19937 rng(std::random_device{}());

      const T std_dev = std::sqrt(T{1} / static_cast<T>(hidden_dim));
      std::normal_distribution<T> dist(T{0}, std_dev);

      for (size_t i = 0; i < w_ih_.size(); ++i) w_ih_.data()[i] = dist(rng);
      for (size_t i = 0; i < w_hh_.size(); ++i) w_hh_.data()[i] = dist(rng);
    }

    Tensor<T> forward(const Tensor<T> &x, const Tensor<T> &h_prev) {
      saved_x_ = x;
      saved_h_prev_ = h_prev;

      // h_t = tanh(W_ih * x + W_hh * h_prev + b)
      Tensor<T> z = w_ih_.matmul(x) + w_hh_.matmul(h_prev) + bias_;
      saved_h_ = z.tanh();
      return saved_h_;
    }

    void backward(const Tensor<T> &grad_h, Tensor<T> &grad_x, Tensor<T> &grad_h_prev) {
      // d/dz tanh(z) = 1 - tanh(z)^2
      Tensor<T> delta = grad_h * (T{1} - saved_h_ * saved_h_);

      Tensor<T> grad_w_ih = Tensor<T>::outer(delta, saved_x_);
      Tensor<T> grad_w_hh = Tensor<T>::outer(delta, saved_h_prev_);

      grad_x = w_ih_.transpose().matmul(delta);
      grad_h_prev = w_hh_.transpose().matmul(delta);

      opt_w_ih_->step(w_ih_, grad_w_ih);
      opt_w_hh_->step(w_hh_, grad_w_hh);
      opt_b_->step(bias_, delta);
    }

    [[nodiscard]] size_t in_dim() const { return in_dim_; }
    [[nodiscard]] size_t hidden_dim() const { return hidden_dim_; }
  };

  // LSTMCell

  template<typename T>
    requires std::is_arithmetic_v<T>
  class LSTMCell final {
    size_t in_dim_;
    size_t hidden_dim_;

    Tensor<T> w_ih_; // (4 * hidden_dim, in_dim)
    Tensor<T> w_hh_; // (4 * hidden_dim, hidden_dim)
    Tensor<T> bias_; // (4 * hidden_dim, 1)

    Tensor<T> saved_x_{{1}};
    Tensor<T> saved_h_prev_{{1}};
    Tensor<T> saved_c_prev_{{1}};

    Tensor<T> saved_f_{{1}};
    Tensor<T> saved_i_{{1}};
    Tensor<T> saved_g_{{1}};
    Tensor<T> saved_o_{{1}};
    Tensor<T> saved_c_{{1}};
    Tensor<T> saved_tanh_c_{{1}};

    std::unique_ptr<Optimizer<T> > opt_w_ih_;
    std::unique_ptr<Optimizer<T> > opt_w_hh_;
    std::unique_ptr<Optimizer<T> > opt_b_;

    static Tensor<T> sigmoid(const Tensor<T> &t) {
      return T{1} / (T{1} + (-t).exp());
    }

  public:
    LSTMCell(size_t in_dim, size_t hidden_dim, const Optimizer<T> &opt_proto)
      : in_dim_(in_dim),
        hidden_dim_(hidden_dim),
        w_ih_({4 * hidden_dim, in_dim}),
        w_hh_({4 * hidden_dim, hidden_dim}),
        bias_({4 * hidden_dim, 1}),
        opt_w_ih_(opt_proto.clone()),
        opt_w_hh_(opt_proto.clone()),
        opt_b_(opt_proto.clone()) {
      bias_.fill(T{0});
      std::mt19937 rng(std::random_device{}());
      const T std_dev = std::sqrt(T{1} / static_cast<T>(hidden_dim));
      std::normal_distribution<T> dist(T{0}, std_dev);

      for (size_t i = 0; i < w_ih_.size(); ++i) w_ih_.data()[i] = dist(rng);
      for (size_t i = 0; i < w_hh_.size(); ++i) w_hh_.data()[i] = dist(rng);
    }

    void forward(const Tensor<T> &x, const Tensor<T> &h_prev, const Tensor<T> &c_prev,
                 Tensor<T> &h_out, Tensor<T> &c_out) {
      saved_x_ = x;
      saved_h_prev_ = h_prev;
      saved_c_prev_ = c_prev;

      Tensor<T> z = w_ih_.matmul(x) + w_hh_.matmul(h_prev) + bias_;

      saved_f_ = sigmoid(z.slice(0, 0 * hidden_dim_, 1 * hidden_dim_));
      saved_i_ = sigmoid(z.slice(0, 1 * hidden_dim_, 2 * hidden_dim_));
      saved_g_ = z.slice(0, 2 * hidden_dim_, 3 * hidden_dim_).tanh();
      saved_o_ = sigmoid(z.slice(0, 3 * hidden_dim_, 4 * hidden_dim_));

      saved_c_ = saved_f_ * saved_c_prev_ + saved_i_ * saved_g_;
      saved_tanh_c_ = saved_c_.tanh();

      h_out = saved_o_ * saved_tanh_c_;
      c_out = saved_c_;
    }

    void backward(const Tensor<T> &grad_h, const Tensor<T> &grad_c,
                  Tensor<T> &grad_x, Tensor<T> &grad_h_prev, Tensor<T> &grad_c_prev) {
      Tensor<T> do_gate = (grad_h * saved_tanh_c_) * saved_o_ * (T{1} - saved_o_);
      Tensor<T> dc_total = grad_c + (grad_h * saved_o_) * (T{1} - saved_tanh_c_ * saved_tanh_c_);

      grad_c_prev = dc_total * saved_f_;

      Tensor<T> df_gate = (dc_total * saved_c_prev_) * saved_f_ * (T{1} - saved_f_);
      Tensor<T> di_gate = (dc_total * saved_g_) * saved_i_ * (T{1} - saved_i_);
      Tensor<T> dg_gate = (dc_total * saved_i_) * (T{1} - saved_g_ * saved_g_);

      Tensor<T> delta({4 * hidden_dim_, 1});
      T *d_ptr = delta.data();

      std::copy_n(df_gate.data(), hidden_dim_, d_ptr + 0 * hidden_dim_);
      std::copy_n(di_gate.data(), hidden_dim_, d_ptr + 1 * hidden_dim_);
      std::copy_n(dg_gate.data(), hidden_dim_, d_ptr + 2 * hidden_dim_);
      std::copy_n(do_gate.data(), hidden_dim_, d_ptr + 3 * hidden_dim_);

      Tensor<T> grad_w_ih = Tensor<T>::outer(delta, saved_x_);
      Tensor<T> grad_w_hh = Tensor<T>::outer(delta, saved_h_prev_);

      grad_x = w_ih_.transpose().matmul(delta);
      grad_h_prev = w_hh_.transpose().matmul(delta);

      opt_w_ih_->step(w_ih_, grad_w_ih);
      opt_w_hh_->step(w_hh_, grad_w_hh);
      opt_b_->step(bias_, delta);
    }

    [[nodiscard]] size_t in_dim() const { return in_dim_; }
    [[nodiscard]] size_t hidden_dim() const { return hidden_dim_; }
  };

  // GRUCell

  template<typename T>
    requires std::is_arithmetic_v<T>
  class GRUCell final {
    size_t in_dim_;
    size_t hidden_dim_;

    Tensor<T> w_ir_; // (hidden_dim, in_dim)
    Tensor<T> w_hr_; // (hidden_dim, hidden_dim)
    Tensor<T> b_r_; // (hidden_dim, 1)

    Tensor<T> w_iz_; // (hidden_dim, in_dim)
    Tensor<T> w_hz_; // (hidden_dim, hidden_dim)
    Tensor<T> b_z_; // (hidden_dim, 1)

    Tensor<T> w_in_; // (hidden_dim, in_dim)
    Tensor<T> w_hn_; // (hidden_dim, hidden_dim)
    Tensor<T> b_in_; // (hidden_dim, 1)
    Tensor<T> b_hn_; // (hidden_dim, 1)

    Tensor<T> saved_x_{{1}};
    Tensor<T> saved_h_prev_{{1}};
    Tensor<T> saved_r_{{1}};
    Tensor<T> saved_z_{{1}};
    Tensor<T> saved_n_{{1}};

    std::unique_ptr<Optimizer<T> > opt_w_ir_, opt_w_hr_, opt_b_r_;
    std::unique_ptr<Optimizer<T> > opt_w_iz_, opt_w_hz_, opt_b_z_;
    std::unique_ptr<Optimizer<T> > opt_w_in_, opt_w_hn_, opt_b_in_, opt_b_hn_;

    static Tensor<T> sigmoid(const Tensor<T> &t) {
      return T{1} / (T{1} + (-t).exp());
    }

  public:
    GRUCell(size_t in_dim, size_t hidden_dim, const Optimizer<T> &opt_proto)
      : in_dim_(in_dim),
        hidden_dim_(hidden_dim),
        w_ir_({hidden_dim, in_dim}), w_hr_({hidden_dim, hidden_dim}), b_r_({hidden_dim, 1}),
        w_iz_({hidden_dim, in_dim}), w_hz_({hidden_dim, hidden_dim}), b_z_({hidden_dim, 1}),
        w_in_({hidden_dim, in_dim}), w_hn_({hidden_dim, hidden_dim}), b_in_({hidden_dim, 1}), b_hn_({hidden_dim, 1}),
        opt_w_ir_(opt_proto.clone()), opt_w_hr_(opt_proto.clone()), opt_b_r_(opt_proto.clone()),
        opt_w_iz_(opt_proto.clone()), opt_w_hz_(opt_proto.clone()), opt_b_z_(opt_proto.clone()),
        opt_w_in_(opt_proto.clone()), opt_w_hn_(opt_proto.clone()), opt_b_in_(opt_proto.clone()),
        opt_b_hn_(opt_proto.clone()) {
      b_r_.fill(T{0});
      b_z_.fill(T{0});
      b_in_.fill(T{0});
      b_hn_.fill(T{0});
      std::mt19937 rng(std::random_device{}());
      const T std_dev = std::sqrt(T{1} / static_cast<T>(hidden_dim));
      std::normal_distribution<T> dist(T{0}, std_dev);

      for (size_t i = 0; i < w_ir_.size(); ++i) {
        w_ir_.data()[i] = dist(rng);
        w_iz_.data()[i] = dist(rng);
        w_in_.data()[i] = dist(rng);
      }
      for (size_t i = 0; i < w_hr_.size(); ++i) {
        w_hr_.data()[i] = dist(rng);
        w_hz_.data()[i] = dist(rng);
        w_hn_.data()[i] = dist(rng);
      }
    }

    Tensor<T> forward(const Tensor<T> &x, const Tensor<T> &h_prev) {
      saved_x_ = x;
      saved_h_prev_ = h_prev;

      saved_r_ = sigmoid(w_ir_.matmul(x) + w_hr_.matmul(h_prev) + b_r_);
      saved_z_ = sigmoid(w_iz_.matmul(x) + w_hz_.matmul(h_prev) + b_z_);

      Tensor<T> linear_hn = w_hn_.matmul(h_prev) + b_hn_;
      saved_n_ = (w_in_.matmul(x) + b_in_ + saved_r_ * linear_hn).tanh();

      return (T{1} - saved_z_) * saved_n_ + saved_z_ * saved_h_prev_;
    }

    void backward(const Tensor<T> &grad_h, Tensor<T> &grad_x, Tensor<T> &grad_h_prev) {
      Tensor<T> dn = grad_h * (T{1} - saved_z_);
      Tensor<T> dz = (grad_h * (saved_h_prev_ - saved_n_)) * saved_z_ * (T{1} - saved_z_);
      Tensor<T> delta_n = dn * (T{1} - saved_n_ * saved_n_);

      Tensor<T> linear_hn = w_hn_.matmul(saved_h_prev_) + b_hn_;
      Tensor<T> delta_r = (delta_n * linear_hn) * saved_r_ * (T{1} - saved_r_);

      Tensor<T> grad_w_in = Tensor<T>::outer(delta_n, saved_x_);
      Tensor<T> grad_w_hn = Tensor<T>::outer(delta_n * saved_r_, saved_h_prev_);
      Tensor<T> grad_b_in = delta_n;
      Tensor<T> grad_b_hn = delta_n * saved_r_;

      Tensor<T> grad_w_iz = Tensor<T>::outer(dz, saved_x_);
      Tensor<T> grad_w_hz = Tensor<T>::outer(dz, saved_h_prev_);
      Tensor<T> grad_b_z = dz;

      Tensor<T> grad_w_ir = Tensor<T>::outer(delta_r, saved_x_);
      Tensor<T> grad_w_hr = Tensor<T>::outer(delta_r, saved_h_prev_);
      Tensor<T> grad_b_r = delta_r;

      grad_x = w_in_.transpose().matmul(delta_n) + w_iz_.transpose().matmul(dz) + w_ir_.transpose().matmul(delta_r);

      grad_h_prev = grad_h * saved_z_
                    + w_hn_.transpose().matmul(delta_n * saved_r_)
                    + w_hz_.transpose().matmul(dz)
                    + w_hr_.transpose().matmul(delta_r);

      opt_w_ir_->step(w_ir_, grad_w_ir);
      opt_w_hr_->step(w_hr_, grad_w_hr);
      opt_b_r_->step(b_r_, grad_b_r);
      opt_w_iz_->step(w_iz_, grad_w_iz);
      opt_w_hz_->step(w_hz_, grad_w_hz);
      opt_b_z_->step(b_z_, grad_b_z);
      opt_w_in_->step(w_in_, grad_w_in);
      opt_w_hn_->step(w_hn_, grad_w_hn);
      opt_b_in_->step(b_in_, grad_b_in);
      opt_b_hn_->step(b_hn_, grad_b_hn);
    }

    [[nodiscard]] size_t in_dim() const { return in_dim_; }
    [[nodiscard]] size_t hidden_dim() const { return hidden_dim_; }
  };

  // PositionalEncoding

  template<typename T>
    requires std::is_arithmetic_v<T>
  class PositionalEncoding final {
    size_t d_model_;

  public:
    explicit PositionalEncoding(size_t d_model)
      : d_model_(d_model) {
    }

    Tensor<T> forward(const Tensor<T> &input) {
      const size_t d_model = input.shape()[0];
      const size_t seq_len = input.shape()[1];

      Tensor<T> output = input.clone();
      T *out_ptr = output.data();

      for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
          const double div_term = std::pow(10000.0, static_cast<double>(2 * (i / 2)) / static_cast<double>(d_model));
          const double val = (i % 2 == 0)
                               ? std::sin(static_cast<double>(pos) / div_term)
                               : std::cos(static_cast<double>(pos) / div_term);

          out_ptr[i * seq_len + pos] += static_cast<T>(val);
        }
      }
      return output;
    }

    Tensor<T> backward(const Tensor<T> &grad_output) {
      return grad_output;
    }
  };

  // SingleHeadAttention

  template<typename T>
    requires std::is_arithmetic_v<T>
  class SingleHeadAttention final {
    size_t d_model_;

    Tensor<T> w_q_; // (d_model, d_model)
    Tensor<T> w_k_; // (d_model, d_model)
    Tensor<T> w_v_; // (d_model, d_model)
    Tensor<T> w_o_; // (d_model, d_model)

    Tensor<T> saved_q_in_{{1}}, saved_k_in_{{1}}, saved_v_in_{{1}};
    Tensor<T> saved_q_{{1}}, saved_k_{{1}}, saved_v_{{1}};
    Tensor<T> saved_attn_{{1}};
    Tensor<T> saved_ctx_{{1}};
    bool causal_mask_ = false;

    std::unique_ptr<Optimizer<T> > opt_q_, opt_k_, opt_v_, opt_o_;

  public:
    SingleHeadAttention(size_t d_model, const Optimizer<T> &opt_proto, bool causal_mask = false)
      : d_model_(d_model),
        w_q_({d_model, d_model}), w_k_({d_model, d_model}),
        w_v_({d_model, d_model}), w_o_({d_model, d_model}),
        causal_mask_(causal_mask),
        opt_q_(opt_proto.clone()), opt_k_(opt_proto.clone()),
        opt_v_(opt_proto.clone()), opt_o_(opt_proto.clone()) {
      std::mt19937 rng(std::random_device{}());
      const T std_dev = std::sqrt(T{1} / static_cast<T>(d_model));
      std::normal_distribution<T> dist(T{0}, std_dev);

      for (size_t i = 0; i < w_q_.size(); ++i) {
        w_q_.data()[i] = dist(rng);
        w_k_.data()[i] = dist(rng);
        w_v_.data()[i] = dist(rng);
        w_o_.data()[i] = dist(rng);
      }
    }

    Tensor<T> forward(const Tensor<T> &q_in, const Tensor<T> &k_in, const Tensor<T> &v_in) {
      saved_q_in_ = q_in;
      saved_k_in_ = k_in;
      saved_v_in_ = v_in;

      const size_t L_q = q_in.shape()[1];
      const size_t L_k = k_in.shape()[1];

      // Projections Q, K, V
      saved_q_ = w_q_.matmul(q_in); // (d_model, L_q)
      saved_k_ = w_k_.matmul(k_in); // (d_model, L_k)
      saved_v_ = w_v_.matmul(v_in); // (d_model, L_k)

      // Raw Attention Scores S = Q^T * K / sqrt(d_model) -> (L_q, L_k)
      const T scale = T{1} / std::sqrt(static_cast<T>(d_model_));
      Tensor<T> scores = saved_q_.transpose().matmul(saved_k_) * scale;

      // Causal Mask (if enabled)
      if (causal_mask_) {
        T *s_ptr = scores.data();
        for (size_t r = 0; r < L_q; ++r) {
          for (size_t c = 0; c < L_k; ++c) {
            if (c > r) s_ptr[r * L_k + c] = T{-10000};
          }
        }
      }

      // Row Softmax
      saved_attn_ = Tensor<T>({L_q, L_k});
      T *a_ptr = saved_attn_.data();
      const T *s_ptr = scores.data();

      for (size_t r = 0; r < L_q; ++r) {
        T max_s = s_ptr[r * L_k];
        for (size_t c = 1; c < L_k; ++c) max_s = std::max(max_s, s_ptr[r * L_k + c]);

        T sum_exp = T{0};
        for (size_t c = 0; c < L_k; ++c) {
          T ev = std::exp(s_ptr[r * L_k + c] - max_s);
          a_ptr[r * L_k + c] = ev;
          sum_exp += ev;
        }
        for (size_t c = 0; c < L_k; ++c) a_ptr[r * L_k + c] /= sum_exp;
      }

      // Context O_ctx = V * A^T -> (d_model, L_q)
      saved_ctx_ = saved_v_.matmul(saved_attn_.transpose());

      // Output Projection Y = W_o * O_ctx -> (d_model, L_q)
      return w_o_.matmul(saved_ctx_);
    }

    void backward(const Tensor<T> &grad_y, Tensor<T> &grad_q_in, Tensor<T> &grad_k_in, Tensor<T> &grad_v_in) {
      const size_t L_q = saved_q_in_.shape()[1];
      const size_t L_k = saved_k_in_.shape()[1];

      // dL/dO_ctx = W_o^T * grad_y
      Tensor<T> grad_ctx = w_o_.transpose().matmul(grad_y);
      Tensor<T> grad_w_o = grad_y.matmul(saved_ctx_.transpose());

      // dL/dV = grad_ctx * A -> (d_model, L_k)
      Tensor<T> grad_v = grad_ctx.matmul(saved_attn_);

      // dL/dA = grad_ctx^T * V -> (L_q, L_k)
      Tensor<T> grad_attn = grad_ctx.transpose().matmul(saved_v_);

      // Softmax Derivative dL/dS -> (L_q, L_k)
      Tensor<T> grad_scores({L_q, L_k});
      const T *a_ptr = saved_attn_.data();
      const T *ga_ptr = grad_attn.data();
      T *gs_ptr = grad_scores.data();

      const T scale = T{1} / std::sqrt(static_cast<T>(d_model_));

      for (size_t r = 0; r < L_q; ++r) {
        T dot = T{0};
        for (size_t c = 0; c < L_k; ++c) {
          const size_t idx = r * L_k + c;
          dot += a_ptr[idx] * ga_ptr[idx];
        }
        for (size_t c = 0; c < L_k; ++c) {
          const size_t idx = r * L_k + c;
          if (causal_mask_ && c > r) {
            gs_ptr[idx] = T{0};
          } else {
            gs_ptr[idx] = a_ptr[idx] * (ga_ptr[idx] - dot) * scale;
          }
        }
      }

      // dL/dQ = K * (dL/dS)^T -> (d_model, L_q)
      Tensor<T> grad_q = saved_k_.matmul(grad_scores.transpose());

      // dL/dK = Q * (dL/dS) -> (d_model, L_k)
      Tensor<T> grad_k = saved_q_.matmul(grad_scores);

      // Parameter Gradients
      Tensor<T> grad_w_q = grad_q.matmul(saved_q_in_.transpose());
      Tensor<T> grad_w_k = grad_k.matmul(saved_k_in_.transpose());
      Tensor<T> grad_w_v = grad_v.matmul(saved_v_in_.transpose());

      // Input Gradients
      grad_q_in = w_q_.transpose().matmul(grad_q);
      grad_k_in = w_k_.transpose().matmul(grad_k);
      grad_v_in = w_v_.transpose().matmul(grad_v);

      opt_q_->step(w_q_, grad_w_q);
      opt_k_->step(w_k_, grad_w_k);
      opt_v_->step(w_v_, grad_w_v);
      opt_o_->step(w_o_, grad_w_o);
    }
  };
} // namespace dl

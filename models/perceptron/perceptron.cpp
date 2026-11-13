#include <functional>
#include <iostream>

#include "dl/dl.h"

float sigmoid(const float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

class Perceptron {
  dl::Tensor<float> weights_;
  float bias_;
  std::function<float(float)> activation_;
  float learning_rate_;
  dl::Tensor<float> saved_input_;

public:
  Perceptron(const size_t input_dim, const std::function<float(float)> &activation, float learning_rate = 0.01f)
    : weights_({input_dim}), bias_(0), activation_(activation), learning_rate_(learning_rate), saved_input_({input_dim}) {
  }

  [[nodiscard]] float forward(const dl::Tensor<float> &input) {
    saved_input_ = input;
    return activation_(weights_.dot(input) + bias_);
  }

  // output_grad: ∂L/∂y
  void backward(const dl::Tensor<float> &output_grad) {
    float g = output_grad[0];

    for (size_t i = 0; i < weights_.size(); ++i) {
      const float grad_w_i = g * saved_input_[i]; // ∂L/∂w_i = ∂L/∂z * x_i
      weights_[i] -= learning_rate_ * grad_w_i;
    }
    bias_ -= learning_rate_ * g; // ∂L/∂b = ∂L/∂z

    // weights_ = weights_ - learning_rate_ * saved_input_.transpose().matmul(output_grad);
    // bias_ = bias_ - learning_rate_ * output_grad.sum();
  }
};

int main() {
  try {
    static float data[6][3] = {
      {0.1, 0.2, 0},
      {0.2, 0.1, 0},
      {0.2, 0.2, 0},
      {0.8, 0.9, 1},
      {0.9, 0.8, 1},
      {0.7, 0.7, 1}
    };

    Perceptron perceptron(2, sigmoid);
    dl::Tensor<float> input({2});

    for (int j = 0; j < 600; j++) {
      const int i = j % 6;
      input.at({0}) = data[i][0];
      input.at({1}) = data[i][1];
      const auto pred  = perceptron.forward(input);
      const auto grad = pred - data[i][2];
      dl::Tensor<float> output_grad({1});
      output_grad.at({0}) = grad;
      perceptron.backward(output_grad);
      // std::cout << "Input: " << *input.data() << " Output: " << pred << std::endl;
    }

    for (int i = 0; i < 6; i++) {
      input.at({0}) = data[i][0];
      input.at({1}) = data[i][1];
      const auto pred  = perceptron.forward(input);
      std::cout << "Input: " << *input.data() << " Output: " << pred << std::endl;
    }
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}

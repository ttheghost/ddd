/**
 * @file ann.cpp
 * @brief ANN on MNIST-784.
 *
 * Architecture
 *   Dense  784 → 128   LeCun-normal init  W ~ N(0, sqrt(1/fan_in))
 *   Sigmoid
 *   Dense  128 → 64   LeCun-normal init
 *   Sigmoid
 *   Dense  64 →  10   LeCun-normal init
 *   Softmax
 */

#include <dl/dl.h>
#include <helper/mnist.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

// Helpers

/** Re-initialize a Dense layer's weights with LeCun normal: N(0, sqrt(1/fan_in)). */
static void lecun_init(dl::Dense<float> &layer, const size_t fan_in,
                       std::mt19937 &rng) {
  const float sd = std::sqrt(1.0f / static_cast<float>(fan_in));
  std::normal_distribution<float> dist(0.f, sd);
  float *w = layer.weights_data();
  for (size_t i = 0; i < layer.weights_size(); ++i) w[i] = dist(rng);
  float *b = layer.bias_data();
  for (size_t i = 0; i < layer.bias_size(); ++i) b[i] = 0.f;
}

/** Cross-entropy loss for one one-hot sample. */
static float cross_entropy(const dl::Tensor<float> &pred,
                           const dl::Tensor<float> &target) {
  float loss = 0.f;
  for (size_t k = 0; k < pred.size(); ++k)
    if (target.data()[k] == 1.f)
      loss -= std::log(std::max(pred.data()[k], 1e-7f));
  return loss;
}

/** Accuracy on a full dataset split. */
static float evaluate(dl::Sequential<float> &model,
                      const std::vector<dl::Tensor<float> > &images,
                      const std::vector<dl::Tensor<float> > &labels) {
  model.set_training(false);
  size_t correct = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    if (model.forward(images[i]).argmax() == labels[i].argmax()) ++correct;
  }
  return static_cast<float>(correct) / static_cast<float>(images.size()) * 100.f;
}

int main() {
  try {
    // Load data
    std::cout << "Loading MNIST...\n";
    auto train_images = load_mnist_images("datasets/mnist/train-images.idx3-ubyte");
    auto train_labels = load_mnist_labels("datasets/mnist/train-labels.idx1-ubyte");
    auto test_images = load_mnist_images("datasets/mnist/t10k-images.idx3-ubyte");
    auto test_labels = load_mnist_labels("datasets/mnist/t10k-labels.idx1-ubyte");
    std::cout << "Loaded " << train_images.size() << " train / "
        << test_images.size() << " test samples.\n\n";

    // Build model
    std::mt19937 init_rng(42);
    dl::SGD<float> sgd(0.1f);

    // Construct all three Dense layers and grab raw pointers before moving
    // into Sequential (unique_ptr ownership transfer).
    auto d1 = std::make_unique<dl::Dense<float> >(784, 128, sgd, false);
    auto d2 = std::make_unique<dl::Dense<float> >(128, 64, sgd, false);
    auto d3 = std::make_unique<dl::Dense<float> >(64, 10, sgd, false);

    // Patch with LeCun-normal
    lecun_init(*d1, 784, init_rng);
    lecun_init(*d2, 128, init_rng);
    lecun_init(*d3, 64, init_rng);

    dl::Sequential<float> model;
    model.add(std::move(d1));
    model.add(std::make_unique<dl::Sigmoid<float> >());
    model.add(std::move(d2));
    model.add(std::make_unique<dl::Sigmoid<float> >());
    model.add(std::move(d3));
    model.add(std::make_unique<dl::Softmax<float> >());

    // Training
    constexpr size_t epochs = 10;
    constexpr size_t print_every = 6000;
    const size_t n_train = train_images.size();

    std::cout << "Architecture : 784 → 128 → Sigmoid → 64 → Sigmoid → 10 → Softmax\n";
    std::cout << "Init         : LeCun normal\n";
    std::cout << "Optimizer    : SGD  lr=0.1\n";
    std::cout << "Samples      : " << n_train << " train  /  "
        << test_images.size() << " test\n";
    std::cout << std::string(72, '-') << "\n";

    std::vector<size_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      std::ranges::shuffle(indices, rng);
      model.set_training(true);

      float total_loss = 0.f;
      size_t correct = 0;
      auto t_start = std::chrono::high_resolution_clock::now();

      for (size_t si = 0; si < n_train; ++si) {
        const auto &x = train_images[indices[si]]; // (784,1)
        const auto &y = train_labels[indices[si]]; // (10,1)  one-hot

        // Forward
        dl::Tensor<float> pred = model.forward(x);

        // Loss + accuracy tracking
        total_loss += cross_entropy(pred, y);
        if (pred.argmax() == y.argmax()) ++correct;

        // Backward — combined CE+Softmax gradient: ∂L/∂z = pred − y
        model.backward(pred - y);

        // Progress bar
        if ((si + 1) % print_every == 0 || si == n_train - 1) {
          const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_start).count();
          const float sps = static_cast<float>(si + 1) / static_cast<float>(elapsed);
          const float loss = total_loss / static_cast<float>(si + 1);
          const float acc = static_cast<float>(correct) /
                            static_cast<float>(si + 1) * 100.f;
          const float pct = static_cast<float>(si + 1) /
                            static_cast<float>(n_train) * 100.f;

          std::cout << "\rEpoch " << std::setw(2) << (epoch + 1)
              << "/" << epochs
              << "  [" << std::setw(5) << (si + 1) << "/" << n_train
              << "  " << std::fixed << std::setprecision(1) << pct << "%]"
              << "  loss=" << std::setprecision(4) << loss
              << "  acc=" << std::setprecision(1) << acc << "%"
              << "  " << std::setprecision(0) << sps << " smp/s"
              << std::flush;
        }
      }

      // End-of-epoch summary + test evaluation
      std::cout << "\n";
      const float train_loss = total_loss / static_cast<float>(n_train);
      const float train_acc = static_cast<float>(correct) /
                              static_cast<float>(n_train) * 100.f;
      const float test_acc = evaluate(model, test_images, test_labels);

      std::cout << "  → train loss=" << std::fixed << std::setprecision(4) << train_loss
          << "  train acc=" << std::setprecision(2) << train_acc << "%"
          << "  test acc=" << std::setprecision(2) << test_acc << "%\n"
          << std::string(72, '-') << "\n";
    }

    std::cout << "\nDone.\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

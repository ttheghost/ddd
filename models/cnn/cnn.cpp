/**
 * @file cnn.cpp
 * @brief Convolutional Neural Network (CNN) trained on MNIST-784.
 *
 * Architecture
 *
 *   Input   : (784, 1) flat MNIST image vector
 *   Reshape : (1, 28, 28)
 *   Conv2D  : 1 → 8 channels, 3×3 kernel, stride=1, padding=1  → (8, 28, 28)
 *   ReLU
 *   MaxPool : 2×2 window, stride=2                            → (8, 14, 14)
 *   Conv2D  : 8 → 16 channels, 3×3 kernel, stride=1, padding=1 → (16, 14, 14)
 *   ReLU
 *   MaxPool : 2×2 window, stride=2                            → (16, 7, 7)
 *   Flatten : (16 * 7 * 7, 1) = (784, 1)
 *   Dense   : 784 → 128 (He init)
 *   ReLU
 *   Dense   : 128 → 10 (He init)
 *   Softmax
 *
 * Optimizer : SGD (lr=0.05)
 * Loss      : Cross-Entropy (combined Softmax+CE gradient ∂L/∂z = pred - y)
 * Training  : 60 000 samples, shuffled per epoch
 * Evaluation: 10 000 test samples after each epoch
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

static float cross_entropy(const dl::Tensor<float> &pred,
                           const dl::Tensor<float> &target) {
  float loss = 0.f;
  for (size_t k = 0; k < pred.size(); ++k)
    if (target.data()[k] == 1.f)
      loss -= std::log(std::max(pred.data()[k], 1e-7f));
  return loss;
}

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
    // Load MNIST Data
    std::cout << "Loading MNIST...\n";
    auto train_images = load_mnist_images("datasets/mnist/train-images.idx3-ubyte");
    auto train_labels = load_mnist_labels("datasets/mnist/train-labels.idx1-ubyte");
    auto test_images = load_mnist_images("datasets/mnist/t10k-images.idx3-ubyte");
    auto test_labels = load_mnist_labels("datasets/mnist/t10k-labels.idx1-ubyte");
    std::cout << "Loaded " << train_images.size() << " train / "
        << test_images.size() << " test samples.\n\n";

    // Build CNN Model
    dl::SGD<float> sgd(0.05f);

    dl::Sequential<float> model;

    // Reshape flat 784 vector into 1×28×28 feature map
    model.add(std::make_unique<dl::Reshape<float> >(std::vector<size_t>{1, 28, 28}));

    // Layer Block 1: Conv(1->8, 3x3) -> ReLU -> MaxPool(2x2)
    model.add(std::make_unique<dl::Conv2D<float> >(1, 8, 3, 1, 1, sgd));
    model.add(std::make_unique<dl::ReLU<float> >());
    model.add(std::make_unique<dl::MaxPool2D<float> >(2, 2));

    // Layer Block 2: Conv(8->16, 3x3) -> ReLU -> MaxPool(2x2)
    model.add(std::make_unique<dl::Conv2D<float> >(8, 16, 3, 1, 1, sgd));
    model.add(std::make_unique<dl::ReLU<float> >());
    model.add(std::make_unique<dl::MaxPool2D<float> >(2, 2));

    // Flatten Block (16 * 7 * 7 = 784)
    model.add(std::make_unique<dl::Flatten<float> >());

    // Fully Connected Classifier: Dense(784->128) -> ReLU -> Dense(128->10) -> Softmax
    model.add(std::make_unique<dl::Dense<float> >(784, 128, sgd));
    model.add(std::make_unique<dl::ReLU<float> >());
    model.add(std::make_unique<dl::Dense<float> >(128, 10, sgd));
    model.add(std::make_unique<dl::Softmax<float> >());

    // Training Loop
    constexpr size_t epochs = 5;
    constexpr size_t print_every = 2000;
    const size_t n_train = train_images.size();

    std::cout <<
        "Architecture : CNN (Reshape -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Dense -> Dense -> Softmax)\n";
    std::cout << "Optimizer    : SGD (lr=0.05)\n";
    std::cout << "Samples      : " << n_train << " train  /  " << test_images.size() << " test\n";
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
        const auto &x = train_images[indices[si]];
        const auto &y = train_labels[indices[si]];

        // Forward
        dl::Tensor<float> pred = model.forward(x);

        // Loss & metrics
        total_loss += cross_entropy(pred, y);
        if (pred.argmax() == y.argmax()) ++correct;

        // Backward: ∂L/∂z = pred - y
        model.backward(pred - y);

        // Live progress display
        if ((si + 1) % print_every == 0 || si == n_train - 1) {
          double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_start).count();
          float sps = static_cast<float>(si + 1) / static_cast<float>(elapsed);
          float loss = total_loss / static_cast<float>(si + 1);
          float acc = static_cast<float>(correct) / static_cast<float>(si + 1) * 100.f;
          float pct = static_cast<float>(si + 1) / static_cast<float>(n_train) * 100.f;

          std::cout << "\rEpoch " << std::setw(2) << (epoch + 1) << "/" << epochs
              << "  [" << std::setw(5) << (si + 1) << "/" << n_train
              << "  " << std::fixed << std::setprecision(1) << pct << "%]"
              << "  loss=" << std::setprecision(4) << loss
              << "  acc=" << std::setprecision(1) << acc << "%"
              << "  " << std::setprecision(0) << sps << " smp/s"
              << std::flush;
        }
      }

      // Epoch evaluation
      std::cout << "\n";
      float train_loss = total_loss / static_cast<float>(n_train);
      float train_acc = static_cast<float>(correct) / static_cast<float>(n_train) * 100.f;
      float test_acc = evaluate(model, test_images, test_labels);

      std::cout << "  → train loss=" << std::fixed << std::setprecision(4) << train_loss
          << "  train acc=" << std::setprecision(2) << train_acc << "%"
          << "  test acc=" << std::setprecision(2) << test_acc << "%\n"
          << std::string(72, '-') << "\n";
    }

    std::cout << "\nCNN Training Complete.\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

#pragma once

#include <dl/tensor.hpp>
#include "utils.hpp"

#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

inline uint32_t swap_endian(const uint32_t val) {
  return ((val << 24) & 0xFF000000) |
         ((val << 8) & 0x00FF0000) |
         ((val >> 8) & 0x0000FF00) |
         ((val >> 24) & 0x000000FF);
}

inline std::vector<dl::Tensor<float> > load_mnist_images(const std::string &filename) {
  std::ifstream file(get_dataset_path(filename), std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  uint32_t magic_number = 0, num_images = 0, rows = 0, cols = 0;
  file.read(reinterpret_cast<char *>(&magic_number), 4);
  file.read(reinterpret_cast<char *>(&num_images), 4);
  file.read(reinterpret_cast<char *>(&rows), 4);
  file.read(reinterpret_cast<char *>(&cols), 4);

  magic_number = swap_endian(magic_number);
  num_images = swap_endian(num_images);
  rows = swap_endian(rows);
  cols = swap_endian(cols);

  const size_t image_size = rows * cols; // 28 * 28 = 784
  std::vector<dl::Tensor<float> > images;
  images.reserve(num_images);

  std::vector<uint8_t> buffer(image_size);
  for (size_t i = 0; i < num_images; ++i) {
    file.read(reinterpret_cast<char *>(buffer.data()), image_size);

    // Normalize pixel values [0, 255] -> [0.0, 1.0] and store as shape (784, 1)
    dl::Tensor<float> img({image_size, 1});
    for (size_t j = 0; j < image_size; ++j) {
      img.data()[j] = static_cast<float>(buffer[j]) / 255.0f;
    }
    images.push_back(std::move(img));
  }
  return images;
}

inline std::vector<dl::Tensor<float> > load_mnist_labels(const std::string &filename) {
  std::ifstream file(get_dataset_path(filename), std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  uint32_t magic_number = 0, num_labels = 0;
  file.read(reinterpret_cast<char *>(&magic_number), 4);
  file.read(reinterpret_cast<char *>(&num_labels), 4);

  magic_number = swap_endian(magic_number);
  num_labels = swap_endian(num_labels);

  std::vector<dl::Tensor<float> > labels;
  labels.reserve(num_labels);

  for (size_t i = 0; i < num_labels; ++i) {
    uint8_t label = 0;
    file.read(reinterpret_cast<char *>(&label), 1);

    // One-hot target vector of shape (10, 1)
    dl::Tensor<float> target({10, 1});
    target.fill(0.0f);
    target.data()[label] = 1.0f;
    labels.push_back(std::move(target));
  }
  return labels;
}

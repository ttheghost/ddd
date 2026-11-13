#pragma once

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

inline std::string get_dataset_path(const std::string &relative_path) {
  fs::path p = relative_path;

  for (int i = 0; i < 4; ++i) {
    if (fs::exists(p)) {
      return p.string();
    }
    p = ".." / p;
  }

  return relative_path;
}

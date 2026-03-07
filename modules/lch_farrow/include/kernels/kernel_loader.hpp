#pragma once

/**
 * @file kernel_loader.hpp
 * @brief Kernel file loader for lch_farrow module
 */

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lch_farrow {

/**
 * Load OpenCL kernel source from .cl file
 * @param filename File name (e.g. "lch_farrow_delay.cl")
 * @return Kernel source as string
 */
inline std::string LoadKernelFile(const std::string& filename) {
  std::string path = std::string(LCH_FARROW_KERNELS_DIR) + "/" + filename;
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error(
        "lch_farrow::LoadKernelFile: cannot open kernel file: " + path);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace lch_farrow

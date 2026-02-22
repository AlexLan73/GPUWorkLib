#pragma once

/**
 * @file kernel_loader.hpp
 * @brief Kernel file loader for heterodyne module
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace heterodyne {

/**
 * Load OpenCL kernel source from .cl file
 * @param filename File name (e.g. "dechirp_multiply.cl")
 * @return Kernel source as string
 */
inline std::string LoadKernelFile(const std::string& filename) {
  std::string path = std::string(HETERODYNE_KERNELS_DIR) + "/" + filename;
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error(
        "heterodyne::LoadKernelFile: cannot open kernel file: " + path);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace heterodyne
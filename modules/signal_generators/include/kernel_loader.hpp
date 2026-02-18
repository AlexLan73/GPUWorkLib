/**
 * @file kernel_loader.hpp
 * @brief Utility for loading OpenCL kernel source from .cl files
 *
 * Reads .cl files from the module's kernels/ directory at runtime.
 * Path is set by CMake via SIGNAL_GENERATORS_KERNELS_DIR.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifndef SIGNAL_GENERATORS_KERNELS_DIR
#define SIGNAL_GENERATORS_KERNELS_DIR "modules/signal_generators/kernels"
#endif

namespace signal_gen {

/**
 * @brief Load OpenCL kernel source from a .cl file
 * @param filename File name (e.g. "form_signal.cl")
 * @return Kernel source as string
 * @throws std::runtime_error if file cannot be opened
 */
inline std::string LoadKernelFile(const std::string& filename) {
  std::string path = std::string(SIGNAL_GENERATORS_KERNELS_DIR) + "/" + filename;
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error(
        "LoadKernelFile: cannot open kernel file: " + path);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/**
 * @brief Load kernel with PRNG prepended
 *
 * Reads prng.cl and concatenates it with the specified kernel file.
 * Used for kernels that require Philox PRNG + Box-Muller.
 *
 * @param kernel_filename Kernel file name (e.g. "form_signal.cl")
 * @return Combined source: prng.cl + kernel
 */
inline std::string LoadKernelWithPrng(const std::string& kernel_filename) {
  return LoadKernelFile("prng.cl") + "\n" + LoadKernelFile(kernel_filename);
}

} // namespace signal_gen

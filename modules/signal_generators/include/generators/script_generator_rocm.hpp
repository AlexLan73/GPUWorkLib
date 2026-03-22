#pragma once

/**
 * @file script_generator_rocm.hpp
 * @brief ScriptGeneratorROCm — text DSL → HIP kernel (hiprtc) compiler
 *
 * ROCm port of ScriptGenerator. Same DSL format:
 *   [Params] ANTENNAS, POINTS
 *   [Defs]   per-antenna variables
 *   [Signal] formula using ID, T, defs
 *
 * Compiles DSL → HIP C++ kernel source → hiprtc → hipModuleLaunchKernel.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "interface/i_backend.hpp"
#include "interface/gpu_context.hpp"

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

struct ScriptParams;   // forward (defined in script_generator.hpp)
struct ParsedScript;   // forward

class ScriptGeneratorROCm {
public:
  explicit ScriptGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~ScriptGeneratorROCm();

  ScriptGeneratorROCm(const ScriptGeneratorROCm&) = delete;
  ScriptGeneratorROCm& operator=(const ScriptGeneratorROCm&) = delete;
  ScriptGeneratorROCm(ScriptGeneratorROCm&& other) noexcept;
  ScriptGeneratorROCm& operator=(ScriptGeneratorROCm&& other) noexcept;

  void LoadScript(const std::string& script_text);
  void LoadFile(const std::string& file_path);

  /// Generate on GPU → void* (caller must hipFree)
  void* Generate();

  /// Generate and read back to CPU
  std::vector<std::complex<float>> GenerateToCpu();

  uint32_t GetAntennas() const;
  uint32_t GetPoints() const;
  size_t GetTotalSamples() const;
  const std::string& GetKernelSource() const { return kernel_source_; }
  bool IsReady() const { return module_ != nullptr; }

private:
  ParsedScript ParseScript(const std::string& text);
  std::string GenerateHIPKernelSource(const ParsedScript& script);
  std::string PrepareExpression(const std::string& line);
  void CompileKernel(const std::string& source);
  void ReleaseGpuResources();

  static std::string Trim(const std::string& s);
  static std::string ToUpper(const std::string& s);

  drv_gpu_lib::IBackend* backend_ = nullptr;
  hipStream_t stream_ = nullptr;

  hipModule_t module_ = nullptr;
  hipFunction_t kernel_fn_ = nullptr;

  uint32_t antennas_ = 0;
  uint32_t points_ = 0;
  std::string kernel_source_;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include "interface/i_backend.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <complex>

namespace signal_gen {
class ScriptGeneratorROCm {
public:
  explicit ScriptGeneratorROCm(drv_gpu_lib::IBackend*) {}
  void LoadScript(const std::string&) { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled"); }
  void* Generate() { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled"); }
  std::vector<std::complex<float>> GenerateToCpu() { throw std::runtime_error("ScriptGeneratorROCm: ROCm not enabled"); }
  bool IsReady() const { return false; }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM

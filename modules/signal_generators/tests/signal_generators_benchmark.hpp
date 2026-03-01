#pragma once

/**
 * @file signal_generators_benchmark.hpp
 * @brief OpenCL benchmark-классы для CwGenerator, LfmGenerator,
 *        LfmConjugateGenerator, NoiseGenerator (GpuBenchmarkBase)
 *
 * CwGeneratorBenchmark         → GenerateToGpu(): "Kernel"
 * LfmGeneratorBenchmark        → GenerateToGpu(): "Kernel"
 * LfmConjugateGeneratorBenchmark→ GenerateToGpu(): "Kernel"
 * NoiseGeneratorBenchmark      → GenerateToGpu(): "Kernel"
 *
 * Использование:
 * @code
 *   signal_gen::CwGenerator gen(backend, cw_params);
 *   CwGeneratorBenchmark bench(backend, gen, system, /*beam_count=*/1);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_signal_generators_profiling.md
 */

#include "../include/generators/cw_generator.hpp"
#include "../include/generators/lfm_generator.hpp"
#include "../include/generators/lfm_conjugate_generator.hpp"
#include "../include/generators/noise_generator.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <CL/cl.h>
#include <cstddef>

namespace test_signal_generators {

// ─── Benchmark 1: CwGenerator::GenerateToGpu() ────────────────────────────

class CwGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на CwGenerator (не владеет, параметры уже установлены)
   * @param system  Параметры дискретизации (fs, length)
   * @param beam_count Количество лучей
   * @param cfg     Параметры бенчмарка
   */
  CwGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::CwGenerator& gen,
      const signal_gen::SystemSampling& system,
      size_t beam_count = 1,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_CwGenerator"})
    : GpuBenchmarkBase(backend, "CwGenerator", cfg),
      gen_(gen), system_(system), beam_count_(beam_count) {}

protected:
  /// Warmup — GenerateToGpu без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_);
    clReleaseMemObject(buf);
  }

  /// Замер — GenerateToGpu с ProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    signal_gen::CwGenerator::ProfEvents pe;
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_, &pe);
    clReleaseMemObject(buf);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::CwGenerator&      gen_;
  signal_gen::SystemSampling    system_;
  size_t                        beam_count_;
};

// ─── Benchmark 2: LfmGenerator::GenerateToGpu() ───────────────────────────

class LfmGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на LfmGenerator (не владеет, параметры уже установлены)
   * @param system  Параметры дискретизации (fs, length)
   * @param beam_count Количество лучей
   * @param cfg     Параметры бенчмарка
   */
  LfmGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::LfmGenerator& gen,
      const signal_gen::SystemSampling& system,
      size_t beam_count = 1,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_LfmGenerator"})
    : GpuBenchmarkBase(backend, "LfmGenerator", cfg),
      gen_(gen), system_(system), beam_count_(beam_count) {}

protected:
  void ExecuteKernel() override {
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_);
    clReleaseMemObject(buf);
  }

  void ExecuteKernelTimed() override {
    signal_gen::LfmGenerator::ProfEvents pe;
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_, &pe);
    clReleaseMemObject(buf);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::LfmGenerator&     gen_;
  signal_gen::SystemSampling    system_;
  size_t                        beam_count_;
};

// ─── Benchmark 3: LfmConjugateGenerator::GenerateToGpu() ─────────────────

class LfmConjugateGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на LfmConjugateGenerator (не владеет;
   *                SetSampling() и SetParams() уже вызваны)
   * @param cfg     Параметры бенчмарка
   */
  LfmConjugateGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::LfmConjugateGenerator& gen,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_LfmConjugateGenerator"})
    : GpuBenchmarkBase(backend, "LfmConjugateGenerator", cfg),
      gen_(gen) {}

protected:
  void ExecuteKernel() override {
    cl_mem buf = gen_.GenerateToGpu();
    clReleaseMemObject(buf);
  }

  void ExecuteKernelTimed() override {
    signal_gen::LfmConjugateGenerator::ProfEvents pe;
    cl_mem buf = gen_.GenerateToGpu(&pe);
    clReleaseMemObject(buf);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::LfmConjugateGenerator& gen_;
};

// ─── Benchmark 4: NoiseGenerator::GenerateToGpu() ─────────────────────────

class NoiseGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на NoiseGenerator (не владеет, параметры уже установлены)
   * @param system  Параметры дискретизации (fs, length)
   * @param beam_count Количество лучей
   * @param cfg     Параметры бенчмарка
   */
  NoiseGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::NoiseGenerator& gen,
      const signal_gen::SystemSampling& system,
      size_t beam_count = 1,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_NoiseGenerator"})
    : GpuBenchmarkBase(backend, "NoiseGenerator", cfg),
      gen_(gen), system_(system), beam_count_(beam_count) {}

protected:
  void ExecuteKernel() override {
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_);
    clReleaseMemObject(buf);
  }

  void ExecuteKernelTimed() override {
    signal_gen::NoiseGenerator::ProfEvents pe;
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_, &pe);
    clReleaseMemObject(buf);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::NoiseGenerator&   gen_;
  signal_gen::SystemSampling    system_;
  size_t                        beam_count_;
};

}  // namespace test_signal_generators

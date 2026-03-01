#pragma once

/**
 * @file fft_maxima_benchmark.hpp
 * @brief Benchmark-классы для SpectrumMaximaFinder (OpenCL, GpuBenchmarkBase)
 *
 * SpectrumMaximaFinderBenchmark    → Process (ONE_PEAK): Upload+FFT+PostKernel
 * SpectrumMaximaAllMaximaBenchmark → FindAllMaxima: Upload+FFT+Detect+Scan+Compact
 *
 * SpectrumMaximaFinder — ЧИСТЫЙ production-класс (ноль кода профилирования).
 * Профилирование работает через опциональный ProfEvents:
 *  - ExecuteKernel()      → Process(input)          — без events (warmup)
 *  - ExecuteKernelTimed() → Process(input, &events) — с events → RecordEvent → GPUProfiler
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_fft_maxima_profiling_opencl.md
 */

#include "spectrum_maxima_finder.h"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <complex>
#include <vector>

namespace test_fft_maxima {

// ─── Benchmark 1: Process (ONE_PEAK) ─────────────────────────────────────────

/**
 * @class SpectrumMaximaFinderBenchmark
 * @brief Benchmark Process(ONE_PEAK): Upload + FFT + PostKernel
 *
 * ExecuteKernel()      → Process() без prof_events — warmup, события освобождаются внутри
 * ExecuteKernelTimed() → Process() с &events → RecordEvent() для каждого cl_event
 */
class SpectrumMaximaFinderBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumMaximaFinderBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumMaximaFinder& proc,
      const antenna_fft::InputData<std::vector<std::complex<float>>>& input,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_Process"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_Process", cfg),
      proc_(proc),
      input_(input) {}

protected:
  /// Warmup — Process без timing (события освобождаются внутри)
  void ExecuteKernel() override {
    proc_.Process(input_);
  }

  /// Замер — Process с ProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    antenna_fft::SpectrumMaximaFinder::ProfEvents events;
    proc_.Process(input_,
                  antenna_fft::PeakSearchMode::ONE_PEAK,
                  antenna_fft::DriverType::OPENCL,
                  &events);
    for (auto& [name, ev] : events)
      RecordEvent(name, ev);
  }

private:
  antenna_fft::SpectrumMaximaFinder&                                    proc_;
  antenna_fft::InputData<std::vector<std::complex<float>>>              input_;
};

// ─── Benchmark 2: FindAllMaxima (full pipeline) ───────────────────────────────

/**
 * @class SpectrumMaximaAllMaximaBenchmark
 * @brief Benchmark FindAllMaxima: Upload + FFT + Detect + Scan + Compact
 *
 * ExecuteKernel()      → FindAllMaxima() без prof_events — warmup
 * ExecuteKernelTimed() → FindAllMaxima() с &events → RecordEvent() → GPUProfiler
 */
class SpectrumMaximaAllMaximaBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumMaximaAllMaximaBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumMaximaFinder& proc,
      const antenna_fft::InputData<std::vector<std::complex<float>>>& input,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_AllMaxima", cfg),
      proc_(proc),
      input_(input) {}

protected:
  /// Warmup — FindAllMaxima без timing
  void ExecuteKernel() override {
    proc_.FindAllMaxima(input_,
                        antenna_fft::OutputDestination::CPU,
                        antenna_fft::DriverType::OPENCL);
  }

  /// Замер — FindAllMaxima с ProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    antenna_fft::SpectrumMaximaFinder::ProfEvents events;
    proc_.FindAllMaxima(input_,
                        antenna_fft::OutputDestination::CPU,
                        antenna_fft::DriverType::OPENCL,
                        0, 0, &events);
    for (auto& [name, ev] : events)
      RecordEvent(name, ev);
  }

private:
  antenna_fft::SpectrumMaximaFinder&                                    proc_;
  antenna_fft::InputData<std::vector<std::complex<float>>>              input_;
};

}  // namespace test_fft_maxima

#pragma once

/**
 * @file filters_benchmark.hpp
 * @brief OpenCL benchmark-классы для FirFilter и IirFilter (GpuBenchmarkBase)
 *
 * FirFilterBenchmark     → Process(cl_mem):  Kernel
 * IirFilterBenchmark     → Process(cl_mem):  Kernel
 *
 * Входные данные (cl_mem) фиксированы между прогонами.
 * Выходной буфер (cl_mem) освобождается после каждого вызова.
 *
 * Использование:
 * @code
 *   FirFilter fir(backend);
 *   fir.SetCoefficients(coeffs);
 *   FirFilterBenchmark bench(backend, fir, input_buf, channels, points);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_filters_profiling.md
 */

#include "filters/fir_filter.hpp"
#include "filters/iir_filter.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <CL/cl.h>
#include <cstdint>

namespace test_filters {

// ─── Benchmark 1: FirFilter::Process(cl_mem) ──────────────────────────────────

class FirFilterBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE) — для GPUProfiler
   * @param filter  Ссылка на FirFilter (не владеет, коэффициенты уже загружены)
   * @param input_buf cl_mem со входными данными (не владеет)
   * @param channels Количество каналов
   * @param points   Количество отсчётов на канал
   * @param cfg      Параметры бенчмарка
   */
  FirFilterBenchmark(
      drv_gpu_lib::IBackend* backend,
      filters::FirFilter& filter,
      cl_mem input_buf,
      uint32_t channels,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_FirFilter"})
    : GpuBenchmarkBase(backend, "FirFilter", cfg),
      filter_(filter), input_buf_(input_buf),
      channels_(channels), points_(points) {}

protected:
  /// Warmup — Process без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    auto result = filter_.Process(input_buf_, channels_, points_);
    clReleaseMemObject(result.data);
  }

  /// Замер — Process с ProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    filters::ProfEvents pe;
    auto result = filter_.Process(input_buf_, channels_, points_, &pe);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
    clReleaseMemObject(result.data);
  }

private:
  filters::FirFilter& filter_;
  cl_mem              input_buf_;
  uint32_t            channels_;
  uint32_t            points_;
};

// ─── Benchmark 2: IirFilter::Process(cl_mem) ──────────────────────────────────

class IirFilterBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE) — для GPUProfiler
   * @param filter  Ссылка на IirFilter (не владеет, секции уже загружены)
   * @param input_buf cl_mem со входными данными (не владеет)
   * @param channels Количество каналов
   * @param points   Количество отсчётов на канал
   * @param cfg      Параметры бенчмарка
   */
  IirFilterBenchmark(
      drv_gpu_lib::IBackend* backend,
      filters::IirFilter& filter,
      cl_mem input_buf,
      uint32_t channels,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_IirFilter"})
    : GpuBenchmarkBase(backend, "IirFilter", cfg),
      filter_(filter), input_buf_(input_buf),
      channels_(channels), points_(points) {}

protected:
  /// Warmup — Process без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    auto result = filter_.Process(input_buf_, channels_, points_);
    clReleaseMemObject(result.data);
  }

  /// Замер — Process с ProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    filters::ProfEvents pe;
    auto result = filter_.Process(input_buf_, channels_, points_, &pe);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
    clReleaseMemObject(result.data);
  }

private:
  filters::IirFilter& filter_;
  cl_mem              input_buf_;
  uint32_t            channels_;
  uint32_t            points_;
};

}  // namespace test_filters

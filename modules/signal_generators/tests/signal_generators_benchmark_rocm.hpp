#pragma once

/**
 * @file signal_generators_benchmark_rocm.hpp
 * @brief ROCm benchmark-класс для FormSignalGeneratorROCm (GpuBenchmarkBase)
 *
 * FormSignalGeneratorROCmBenchmark → GenerateInputData(): "Kernel"
 *
 * Компилируется только при ENABLE_ROCM=1 (Linux + AMD GPU).
 * На Windows без AMD GPU: compile-only, не выполняется.
 *
 * Использование:
 * @code
 *   FormSignalGeneratorROCm gen(backend);
 *   gen.SetParams(params);
 *   FormSignalGeneratorROCmBenchmark bench(backend, gen);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_signal_generators_profiling.md
 */

#if ENABLE_ROCM

#include "../include/generators/form_signal_generator_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <hip/hip_runtime.h>

namespace test_signal_generators_rocm {

// ─── Benchmark: FormSignalGeneratorROCm::GenerateInputData() ──────────────

class FormSignalGeneratorROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (ROCm) — для GPUProfiler
   * @param gen     Ссылка на FormSignalGeneratorROCm (не владеет,
   *                SetParams() уже вызван)
   * @param cfg     Параметры бенчмарка
   */
  FormSignalGeneratorROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::FormSignalGeneratorROCm& gen,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_FormSignalROCm"})
    : GpuBenchmarkBase(backend, "FormSignalROCm", cfg),
      gen_(gen) {}

protected:
  /// Warmup — GenerateInputData без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    auto input = gen_.GenerateInputData();
    hipFree(input.data);
  }

  /// Замер — GenerateInputData с ROCmProfEvents → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    signal_gen::ROCmProfEvents events;
    auto input = gen_.GenerateInputData(&events);
    hipFree(input.data);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  signal_gen::FormSignalGeneratorROCm& gen_;
};

}  // namespace test_signal_generators_rocm

#endif  // ENABLE_ROCM

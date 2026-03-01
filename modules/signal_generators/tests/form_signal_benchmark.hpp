#pragma once

/**
 * @file form_signal_benchmark.hpp
 * @brief OpenCL benchmark-классы для Form генераторов (GpuBenchmarkBase)
 *
 * FormSignalGeneratorBenchmark      → GenerateInputData(): "Kernel"
 * DelayedFormSignalGeneratorBenchmark→ GenerateInputData(): "FormSignal" + "FarrowDelay"
 * LfmAnalyticalDelayBenchmark       → GenerateToGpu():     "Kernel"
 * FormScriptGeneratorBenchmark      → GenerateInputData(): "Kernel"
 *
 * Замечания:
 * - Все классы не владеют генераторами (передаются по ссылке)
 * - Для DelayedFormSignalGenerator события именуются "FormSignal" и "FarrowDelay"
 * - Для FormScriptGenerator kernel должен быть скомпилирован до создания бенчмарка
 * - Для LfmAnalyticalDelayBenchmark SetDelays() должен быть вызван до создания бенчмарка
 *
 * Использование:
 * @code
 *   signal_gen::FormSignalGenerator gen(backend);
 *   gen.SetParams(params);
 *   FormSignalGeneratorBenchmark bench(backend, gen);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_signal_generators_profiling.md
 */

#include "../include/generators/form_signal_generator.hpp"
#include "../include/generators/delayed_form_signal_generator.hpp"
#include "../include/generators/lfm_generator_analytical_delay.hpp"
#include "../include/generators/form_script_generator.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <CL/cl.h>

namespace test_form_signal {

// ─── Benchmark 1: FormSignalGenerator::GenerateInputData() ────────────────

class FormSignalGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на FormSignalGenerator (не владеет, SetParams() уже вызван)
   * @param cfg     Параметры бенчмарка
   */
  FormSignalGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::FormSignalGenerator& gen,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_FormSignal"})
    : GpuBenchmarkBase(backend, "FormSignalGenerator", cfg),
      gen_(gen) {}

protected:
  /// Warmup — GenerateInputData без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    auto input = gen_.GenerateInputData();
    clReleaseMemObject(input.data);
  }

  /// Замер — GenerateInputData с ProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    signal_gen::FormSignalGenerator::ProfEvents pe;
    auto input = gen_.GenerateInputData(&pe);
    clReleaseMemObject(input.data);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::FormSignalGenerator& gen_;
};

// ─── Benchmark 2: DelayedFormSignalGenerator::GenerateInputData() ─────────

class DelayedFormSignalGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на DelayedFormSignalGenerator (не владеет;
   *                SetParams() и SetDelays() уже вызваны)
   * @param cfg     Параметры бенчмарка
   *
   * События: "FormSignal" (form_signal.cl) + "FarrowDelay" (delayed_form_signal.cl)
   */
  DelayedFormSignalGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::DelayedFormSignalGenerator& gen,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_DelayedFormSignal"})
    : GpuBenchmarkBase(backend, "DelayedFormSignalGenerator", cfg),
      gen_(gen) {}

protected:
  void ExecuteKernel() override {
    auto input = gen_.GenerateInputData();
    clReleaseMemObject(input.data);
  }

  void ExecuteKernelTimed() override {
    signal_gen::DelayedFormSignalGenerator::ProfEvents pe;
    auto input = gen_.GenerateInputData(&pe);
    clReleaseMemObject(input.data);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::DelayedFormSignalGenerator& gen_;
};

// ─── Benchmark 3: LfmGeneratorAnalyticalDelay::GenerateToGpu() ────────────

class LfmAnalyticalDelayBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на LfmGeneratorAnalyticalDelay (не владеет;
   *                SetParams(), SetSampling(), SetDelays() уже вызваны)
   * @param cfg     Параметры бенчмарка
   */
  LfmAnalyticalDelayBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::LfmGeneratorAnalyticalDelay& gen,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_LfmAnalyticalDelay"})
    : GpuBenchmarkBase(backend, "LfmAnalyticalDelay", cfg),
      gen_(gen) {}

protected:
  void ExecuteKernel() override {
    auto result = gen_.GenerateToGpu();
    clReleaseMemObject(result.data);
  }

  void ExecuteKernelTimed() override {
    signal_gen::LfmGeneratorAnalyticalDelay::ProfEvents pe;
    auto result = gen_.GenerateToGpu(&pe);
    clReleaseMemObject(result.data);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::LfmGeneratorAnalyticalDelay& gen_;
};

// ─── Benchmark 4: FormScriptGenerator::GenerateInputData() ────────────────

class FormScriptGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (OpenCL, с CL_QUEUE_PROFILING_ENABLE)
   * @param gen     Ссылка на FormScriptGenerator (не владеет;
   *                Compile() или LoadKernel() уже вызван — IsReady() == true)
   * @param cfg     Параметры бенчмарка
   */
  FormScriptGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::FormScriptGenerator& gen,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_FormScriptGenerator"})
    : GpuBenchmarkBase(backend, "FormScriptGenerator", cfg),
      gen_(gen) {}

protected:
  void ExecuteKernel() override {
    auto input = gen_.GenerateInputData();
    clReleaseMemObject(input.data);
  }

  void ExecuteKernelTimed() override {
    signal_gen::FormScriptGenerator::ProfEvents pe;
    auto input = gen_.GenerateInputData(&pe);
    clReleaseMemObject(input.data);
    for (auto& [name, ev] : pe)
      RecordEvent(name, ev);
  }

private:
  signal_gen::FormScriptGenerator& gen_;
};

}  // namespace test_form_signal

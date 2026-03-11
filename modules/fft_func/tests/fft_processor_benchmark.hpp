#pragma once

/**
 * @file fft_processor_benchmark.hpp
 * @brief FFTProcessorBenchmark — наследник GpuBenchmarkBase для FFTProcessor
 *
 * Пример использования:
 * @code
 *   FFTProcessor proc(backend);
 *   FFTProcessorBenchmark bench(backend, proc, params, input_data);
 *   bench.Run();     // warmup(5) + measure(20) → GPUProfiler
 *   bench.Report();  // profiler.PrintReport() + ExportJSON + ExportMarkdown
 * @endcode
 *
 * FFTProcessor — ЧИСТЫЙ production-класс (ноль кода профилирования).
 * Профилирование работает через опциональный prof_events:
 *  - ExecuteKernel()      → ProcessComplex(data, params)       — без events
 *  - ExecuteKernelTimed() → ProcessComplex(data, params, &ev)  — с events
 *    → RecordEvent() для каждого cl_event → GPUProfiler
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/specs/Profil_GPU.md
 */

#include "fft_processor.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <complex>
#include <vector>
#include <utility>

namespace test_fft_processor {

class FFTProcessorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend для инициализации GPUProfiler
   * @param proc Ссылка на чистый FFTProcessor (не владеет)
   * @param params Параметры FFT (фиксированы на весь бенчмарк)
   * @param input_data Входные данные (фиксированы — не меняются между запусками)
   * @param cfg Параметры бенчмарка (n_warmup=5, n_runs=20)
   */
  FFTProcessorBenchmark(
      drv_gpu_lib::IBackend* backend,
      fft_processor::FFTProcessor& proc,
      const fft_processor::FFTProcessorParams& params,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_FFT"})
    : GpuBenchmarkBase(backend, "FFTProcessor", cfg),
      proc_(proc),
      params_(params),
      input_data_(input_data) {}

protected:
  /**
   * @brief Warmup — запуск FFT БЕЗ timing
   *
   * ProcessComplex() без prof_events → события освобождаются внутри.
   * Ноль overhead. Просто прогрев GPU (JIT, clock ramp-up, shader cache).
   */
  void ExecuteKernel() override {
    proc_.ProcessComplex(input_data_, params_);
  }

  /**
   * @brief Замер — запуск FFT С timing → RecordEvent → GPUProfiler
   *
   * ProcessComplex() с prof_events → собирает cl_event'ы.
   * Каждый cl_event (Upload, FFT, Download) записывается отдельно
   * в GPUProfiler через RecordEvent() из GpuBenchmarkBase.
   *
   * GPUProfiler копит все вызовы → min/max/avg автоматически.
   */
  void ExecuteKernelTimed() override {
    std::vector<std::pair<const char*, cl_event>> events;
    proc_.ProcessComplex(input_data_, params_, &events);

    // Записать каждый cl_event в GPUProfiler:
    // RecordEvent(): clWaitForEvents + FillOpenCLProfilingData + profiler.Record + clReleaseEvent
    for (auto& [name, ev] : events) {
      RecordEvent(name, ev);
    }
  }

private:
  fft_processor::FFTProcessor&      proc_;
  fft_processor::FFTProcessorParams params_;
  std::vector<std::complex<float>>  input_data_;
};

}  // namespace test_fft_processor

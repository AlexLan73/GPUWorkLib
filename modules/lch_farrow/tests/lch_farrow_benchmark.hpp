#pragma once

/**
 * @file lch_farrow_benchmark.hpp
 * @brief LchFarrowBenchmark — наследник GpuBenchmarkBase для LchFarrow (OpenCL)
 *
 * LchFarrow — ЧИСТЫЙ production-класс (ноль кода профилирования).
 * Профилирование через опциональный prof_events:
 *  - ExecuteKernel()      → Process(input_buf_, ...) — без событий (warmup)
 *  - ExecuteKernelTimed() → Process(input_buf_, ..., &events) — с cl_event
 *    → RecordEvent() для каждого события → GPUProfiler
 *
 * Stages:
 *  - Upload_delay : clEnqueueWriteBuffer (delay_us на GPU, каждый вызов)
 *  - Kernel       : lch_farrow_delay (Lagrange 48x5 интерполяция)
 *
 * input_buf загружается ОДИН РАЗ в test runner (не входит в замер).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, Doc_Addition/GPU_Profiling_Mechanism.md
 */

#include "lch_farrow.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <CL/cl.h>
#include <vector>
#include <utility>
#include <cstdint>

namespace test_lch_farrow {

class LchFarrowBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @param backend   IBackend для GPUProfiler
   * @param proc      Ссылка на LchFarrow (не владеет)
   * @param input_buf cl_mem с входным сигналом на GPU (не освобождается здесь)
   * @param antennas  Число антенн
   * @param points    Число отсчётов на антенну
   * @param cfg       Параметры бенчмарка (n_warmup, n_runs, output_dir)
   */
  LchFarrowBenchmark(
      drv_gpu_lib::IBackend* backend,
      lch_farrow::LchFarrow& proc,
      cl_mem input_buf,
      uint32_t antennas,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_LchFarrow"})
    : GpuBenchmarkBase(backend, "LchFarrow", cfg),
      proc_(proc),
      input_buf_(input_buf),
      antennas_(antennas),
      points_(points) {}

protected:
  /**
   * @brief Warmup — запуск БЕЗ timing (прогрев GPU: JIT, clock ramp-up)
   *
   * proc_.Process() без prof_events → ноль overhead.
   * result.data (output GPU buffer) освобождается сразу.
   */
  void ExecuteKernel() override {
    auto result = proc_.Process(input_buf_, antennas_, points_);
    if (result.data) clReleaseMemObject(result.data);
  }

  /**
   * @brief Замер — запуск С timing → RecordEvent → GPUProfiler
   *
   * proc_.Process() с prof_events → собирает cl_event'ы:
   *   Upload_delay (clEnqueueWriteBuffer), Kernel (lch_farrow_delay)
   * Каждый cl_event записывается через RecordEvent() из GpuBenchmarkBase.
   * GPUProfiler копит все вызовы → min/max/avg автоматически.
   */
  void ExecuteKernelTimed() override {
    lch_farrow::ProfEvents events;
    auto result = proc_.Process(input_buf_, antennas_, points_, &events);
    if (result.data) clReleaseMemObject(result.data);

    for (auto& [name, ev] : events) {
      RecordEvent(name, ev);
    }
  }

private:
  lch_farrow::LchFarrow& proc_;
  cl_mem    input_buf_;
  uint32_t  antennas_;
  uint32_t  points_;
};

}  // namespace test_lch_farrow

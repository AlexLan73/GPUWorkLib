#pragma once

/**
 * @file signal_service.hpp
 * @brief SignalService — Facade для генерации сигналов
 *
 * Единая точка входа: принимает SignalRequest, возвращает данные CPU/GPU.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "signal_generator_factory.hpp"
#include "params/signal_request.hpp"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>

namespace signal_gen {

/**
 * @class SignalService
 * @brief Facade для генерации сигналов на CPU/GPU
 *
 * @code
 * signal_gen::SignalService service(backend);
 *
 * signal_gen::CwParams cw;
 * cw.f0 = 100.0;
 * cw.freq_step = 10.0;
 *
 * // CPU
 * auto cpu_data = service.GenerateCpu(cw, {1000.0, 4096});
 *
 * // GPU
 * cl_mem gpu_data = service.GenerateGpu(cw, {1000.0, 4096}, 8);
 * // ... использовать gpu_data ...
 * clReleaseMemObject(gpu_data);
 * @endcode
 */
class SignalService {
public:
    explicit SignalService(drv_gpu_lib::IBackend* backend)
        : backend_(backend) {}

    // ═══════════════════════════════════════════════════════════════════
    // CPU generation
    // ═══════════════════════════════════════════════════════════════════

    /// Генерация CW на CPU (1 луч)
    std::vector<std::complex<float>> GenerateCpu(
        const CwParams& params, const SystemSampling& system);

    /// Генерация LFM на CPU (1 луч)
    std::vector<std::complex<float>> GenerateCpu(
        const LfmParams& params, const SystemSampling& system);

    /// Генерация Noise на CPU (1 луч)
    std::vector<std::complex<float>> GenerateCpu(
        const NoiseParams& params, const SystemSampling& system);

    // ═══════════════════════════════════════════════════════════════════
    // GPU generation
    // ═══════════════════════════════════════════════════════════════════

    /// Генерация CW на GPU (N лучей)
    cl_mem GenerateGpu(const CwParams& params, const SystemSampling& system,
                       size_t beam_count = 1);

    /// Генерация LFM на GPU (N лучей)
    cl_mem GenerateGpu(const LfmParams& params, const SystemSampling& system,
                       size_t beam_count = 1);

    /// Генерация Noise на GPU (N лучей)
    cl_mem GenerateGpu(const NoiseParams& params, const SystemSampling& system,
                       size_t beam_count = 1);

private:
    drv_gpu_lib::IBackend* backend_;
};

} // namespace signal_gen

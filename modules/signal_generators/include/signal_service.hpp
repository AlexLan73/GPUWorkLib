#pragma once

#if !ENABLE_ROCM  // OpenCL-only facade: ROCm modules use generators directly

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
#include "interface/input_data.hpp"
#include "generators/cw_generator.hpp"
#include "generators/lfm_generator.hpp"
#include "generators/noise_generator.hpp"
#include "generators/form_signal_generator.hpp"
#include "generators/delayed_form_signal_generator.hpp"

#include <CL/cl.h>
#include <memory>
#include <optional>
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

    // ═══════════════════════════════════════════════════════════════════
    // FormSignal generation (standalone API)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @brief Генерация FormSignal на GPU (мультиканальная)
     * @param params FormParams (fs, antennas, points уже внутри)
     * @return InputData<cl_mem> — совместимо с fft_func (data, antenna_count, n_point, gpu_memory_bytes)
     * @note Вызывающий код должен освободить result.data через clReleaseMemObject()
     */
    drv_gpu_lib::InputData<cl_mem> GenerateFormGpu(const FormParams& params);

    /**
     * @brief Генерация FormSignal на CPU (по каналам)
     * @param params FormParams
     * @return vector[antenna_id][sample_id] complex<float>
     */
    std::vector<std::vector<std::complex<float>>> GenerateFormCpu(
        const FormParams& params);

    // ═══════════════════════════════════════════════════════════════════
    // DelayedFormSignal generation (Farrow 48×5)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @brief Генерация сигнала с дробной задержкой (Farrow) на GPU
     * @param params FormParams (fs, antennas, points, noise_amplitude)
     * @param delay_us Задержки per-antenna в микросекундах
     * @return InputData<cl_mem>
     * @note Вызывающий код должен освободить result.data через clReleaseMemObject()
     */
    drv_gpu_lib::InputData<cl_mem> GenerateDelayedFormGpu(
        const FormParams& params, const std::vector<float>& delay_us);

    /**
     * @brief Генерация сигнала с дробной задержкой (Farrow) на CPU
     * @param params FormParams
     * @param delay_us Задержки per-antenna в микросекундах
     * @return vector[antenna_id][sample_id] complex<float>
     */
    std::vector<std::vector<std::complex<float>>> GenerateDelayedFormCpu(
        const FormParams& params, const std::vector<float>& delay_us);

private:
    drv_gpu_lib::IBackend* backend_;

    // Кешированные генераторы: перекомпиляция ядра только при смене params
    std::optional<CwGenerator>                  cw_gen_;
    CwParams                                    cw_params_{};
    std::optional<LfmGenerator>                 lfm_gen_;
    LfmParams                                   lfm_params_{};
    std::optional<NoiseGenerator>               noise_gen_;
    NoiseParams                                 noise_params_{};
    std::optional<FormSignalGenerator>          form_gen_;
    std::optional<DelayedFormSignalGenerator>   delayed_gen_;

    CwGenerator&    GetCw(const CwParams& p);
    LfmGenerator&   GetLfm(const LfmParams& p);
    NoiseGenerator& GetNoise(const NoiseParams& p);
};

} // namespace signal_gen

#endif  // !ENABLE_ROCM

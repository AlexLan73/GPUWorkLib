#pragma once

/**
 * @file cw_generator.hpp
 * @brief CW (Continuous Wave) генератор — синусоида на GPU/CPU
 *
 * Генерирует: s(t) = A * exp(j * (2*pi*f*t + phase))
 * Для multi-beam: freq_i = f0 + i * freq_step
 *
 * Мигрировано из test_signal_generator.hpp + расширено:
 * - IBackend* вместо raw cl_context/queue
 * - Configurable amplitude, phase, freq_step
 * - CPU reference implementation
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "../i_signal_generator.hpp"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <string>
#include <cstring>
#include <utility>
#include <vector>

namespace signal_gen {

class CwGenerator : public ISignalGenerator {
public:
    /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
    using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

    /**
     * @brief Конструктор
     * @param backend GPU backend (не владеет!)
     * @param params  Параметры CW сигнала
     */
    CwGenerator(drv_gpu_lib::IBackend* backend, const CwParams& params);
    ~CwGenerator() override;

    // Запрет копирования
    CwGenerator(const CwGenerator&) = delete;
    CwGenerator& operator=(const CwGenerator&) = delete;

    // Перемещение
    CwGenerator(CwGenerator&& other) noexcept;
    CwGenerator& operator=(CwGenerator&& other) noexcept;

    // ISignalGenerator
    void GenerateToCpu(const SystemSampling& system,
                       std::complex<float>* out, size_t out_size) override;

    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count = 1) override;

    /**
     * @brief Генерация на GPU с опциональным сбором событий профилирования
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *
     * Собирает события: "Kernel" (cw_kernel)
     */
    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count,
                         ProfEvents* prof_events);

    SignalKind Kind() const override { return SignalKind::CW; }

    /// Обновить параметры CW
    void SetParams(const CwParams& params) { params_ = params; }
    const CwParams& GetParams() const { return params_; }

private:
    void CompileKernel();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    CwParams params_;

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
};

} // namespace signal_gen

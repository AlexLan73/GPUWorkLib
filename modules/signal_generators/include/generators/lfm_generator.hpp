#pragma once

/**
 * @file lfm_generator.hpp
 * @brief LFM (Linear Frequency Modulation) генератор — chirp на GPU/CPU
 *
 * Генерирует: s(t) = A * exp(j * pi * k * t^2 + j * 2*pi*f_start*t)
 * где k = (f_end - f_start) / duration — скорость изменения частоты
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "../i_signal_generator.hpp"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <cstring>
#include <utility>
#include <vector>

namespace signal_gen {

class LfmGenerator : public ISignalGenerator {
public:
    /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
    using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

    LfmGenerator(drv_gpu_lib::IBackend* backend, const LfmParams& params);
    ~LfmGenerator() override;

    LfmGenerator(const LfmGenerator&) = delete;
    LfmGenerator& operator=(const LfmGenerator&) = delete;
    LfmGenerator(LfmGenerator&& other) noexcept;
    LfmGenerator& operator=(LfmGenerator&& other) noexcept;

    void GenerateToCpu(const SystemSampling& system,
                       std::complex<float>* out, size_t out_size) override;

    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count = 1) override;

    /**
     * @brief Генерация на GPU с опциональным сбором событий профилирования
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *
     * Собирает события: "Kernel" (lfm_kernel)
     */
    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count,
                         ProfEvents* prof_events);

    SignalKind Kind() const override { return SignalKind::LFM; }

    void SetParams(const LfmParams& params) { params_ = params; }
    const LfmParams& GetParams() const { return params_; }

private:
    void CompileKernel();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    LfmParams params_;

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
};

} // namespace signal_gen

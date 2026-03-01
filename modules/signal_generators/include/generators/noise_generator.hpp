#pragma once

/**
 * @file noise_generator.hpp
 * @brief Noise генератор — Gaussian/White шум на GPU/CPU
 *
 * GPU: Philox-2x32 PRNG + Box-Muller transform
 * CPU: std::mt19937 + std::normal_distribution
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

class NoiseGenerator : public ISignalGenerator {
public:
    /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
    using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

    NoiseGenerator(drv_gpu_lib::IBackend* backend, const NoiseParams& params);
    ~NoiseGenerator() override;

    NoiseGenerator(const NoiseGenerator&) = delete;
    NoiseGenerator& operator=(const NoiseGenerator&) = delete;
    NoiseGenerator(NoiseGenerator&& other) noexcept;
    NoiseGenerator& operator=(NoiseGenerator&& other) noexcept;

    void GenerateToCpu(const SystemSampling& system,
                       std::complex<float>* out, size_t out_size) override;

    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count = 1) override;

    /**
     * @brief Генерация на GPU с опциональным сбором событий профилирования
     * @param prof_events nullptr → production (zero overhead); &vec → benchmark
     *
     * Собирает события: "Kernel" (noise_kernel / Philox+BoxMuller)
     */
    cl_mem GenerateToGpu(const SystemSampling& system,
                         size_t beam_count,
                         ProfEvents* prof_events);

    SignalKind Kind() const override { return SignalKind::NOISE; }

    void SetParams(const NoiseParams& params) { params_ = params; }
    const NoiseParams& GetParams() const { return params_; }

private:
    void CompileKernel();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    NoiseParams params_;

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
};

} // namespace signal_gen

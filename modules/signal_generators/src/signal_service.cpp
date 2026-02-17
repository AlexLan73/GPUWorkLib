/**
 * @file signal_service.cpp
 * @brief Реализация SignalService — Facade для генерации
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "signal_service.hpp"
#include "generators/cw_generator.hpp"
#include "generators/lfm_generator.hpp"
#include "generators/noise_generator.hpp"
#include "generators/form_signal_generator.hpp"
#include "generators/delayed_form_signal_generator.hpp"

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// CPU generation
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> SignalService::GenerateCpu(
    const CwParams& params, const SystemSampling& system)
{
    CwGenerator gen(backend_, params);
    std::vector<std::complex<float>> out(system.length);
    gen.GenerateToCpu(system, out.data(), out.size());
    return out;
}

std::vector<std::complex<float>> SignalService::GenerateCpu(
    const LfmParams& params, const SystemSampling& system)
{
    LfmGenerator gen(backend_, params);
    std::vector<std::complex<float>> out(system.length);
    gen.GenerateToCpu(system, out.data(), out.size());
    return out;
}

std::vector<std::complex<float>> SignalService::GenerateCpu(
    const NoiseParams& params, const SystemSampling& system)
{
    NoiseGenerator gen(backend_, params);
    std::vector<std::complex<float>> out(system.length);
    gen.GenerateToCpu(system, out.data(), out.size());
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU generation
// ════════════════════════════════════════════════════════════════════════════

cl_mem SignalService::GenerateGpu(
    const CwParams& params, const SystemSampling& system, size_t beam_count)
{
    CwGenerator gen(backend_, params);
    return gen.GenerateToGpu(system, beam_count);
}

cl_mem SignalService::GenerateGpu(
    const LfmParams& params, const SystemSampling& system, size_t beam_count)
{
    LfmGenerator gen(backend_, params);
    return gen.GenerateToGpu(system, beam_count);
}

cl_mem SignalService::GenerateGpu(
    const NoiseParams& params, const SystemSampling& system, size_t beam_count)
{
    NoiseGenerator gen(backend_, params);
    return gen.GenerateToGpu(system, beam_count);
}

// ════════════════════════════════════════════════════════════════════════════
// FormSignal generation
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem> SignalService::GenerateFormGpu(const FormParams& params) {
    FormSignalGenerator gen(backend_);
    gen.SetParams(params);
    return gen.GenerateInputData();
}

std::vector<std::vector<std::complex<float>>> SignalService::GenerateFormCpu(
    const FormParams& params) {
    FormSignalGenerator gen(backend_);
    gen.SetParams(params);
    return gen.GenerateToCpu();
}

// ════════════════════════════════════════════════════════════════════════════
// DelayedFormSignal generation (Farrow 48×5)
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem> SignalService::GenerateDelayedFormGpu(
    const FormParams& params, const std::vector<float>& delay_us) {
    DelayedFormSignalGenerator gen(backend_);
    gen.SetParams(params);
    gen.SetDelays(delay_us);
    return gen.GenerateInputData();
}

std::vector<std::vector<std::complex<float>>> SignalService::GenerateDelayedFormCpu(
    const FormParams& params, const std::vector<float>& delay_us) {
    DelayedFormSignalGenerator gen(backend_);
    gen.SetParams(params);
    gen.SetDelays(delay_us);
    return gen.GenerateToCpu();
}

} // namespace signal_gen

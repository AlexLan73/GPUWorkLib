/**
 * @file signal_service.cpp
 * @brief Реализация SignalService — Facade для генерации
 *
 * Генераторы кешируются: ядро компилируется один раз при первом вызове
 * или при смене параметров сигнала (CwParams, LfmParams, NoiseParams).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "signal_service.hpp"

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Lazy-init helpers: пересоздаём генератор только при смене params
// ════════════════════════════════════════════════════════════════════════════

CwGenerator& SignalService::GetCw(const CwParams& p) {
    if (!cw_gen_ || cw_params_ != p) {
        cw_gen_.emplace(backend_, p);
        cw_params_ = p;
    }
    return *cw_gen_;
}

LfmGenerator& SignalService::GetLfm(const LfmParams& p) {
    if (!lfm_gen_ || lfm_params_ != p) {
        lfm_gen_.emplace(backend_, p);
        lfm_params_ = p;
    }
    return *lfm_gen_;
}

NoiseGenerator& SignalService::GetNoise(const NoiseParams& p) {
    if (!noise_gen_ || noise_params_ != p) {
        noise_gen_.emplace(backend_, p);
        noise_params_ = p;
    }
    return *noise_gen_;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU generation
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> SignalService::GenerateCpu(
    const CwParams& params, const SystemSampling& system)
{
    std::vector<std::complex<float>> out(system.length);
    GetCw(params).GenerateToCpu(system, out.data(), out.size());
    return out;
}

std::vector<std::complex<float>> SignalService::GenerateCpu(
    const LfmParams& params, const SystemSampling& system)
{
    std::vector<std::complex<float>> out(system.length);
    GetLfm(params).GenerateToCpu(system, out.data(), out.size());
    return out;
}

std::vector<std::complex<float>> SignalService::GenerateCpu(
    const NoiseParams& params, const SystemSampling& system)
{
    std::vector<std::complex<float>> out(system.length);
    GetNoise(params).GenerateToCpu(system, out.data(), out.size());
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU generation
// ════════════════════════════════════════════════════════════════════════════

cl_mem SignalService::GenerateGpu(
    const CwParams& params, const SystemSampling& system, size_t beam_count)
{
    return GetCw(params).GenerateToGpu(system, beam_count);
}

cl_mem SignalService::GenerateGpu(
    const LfmParams& params, const SystemSampling& system, size_t beam_count)
{
    return GetLfm(params).GenerateToGpu(system, beam_count);
}

cl_mem SignalService::GenerateGpu(
    const NoiseParams& params, const SystemSampling& system, size_t beam_count)
{
    return GetNoise(params).GenerateToGpu(system, beam_count);
}

// ════════════════════════════════════════════════════════════════════════════
// FormSignal generation
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem> SignalService::GenerateFormGpu(const FormParams& params) {
    if (!form_gen_)
        form_gen_.emplace(backend_);
    form_gen_->SetParams(params);
    return form_gen_->GenerateInputData();
}

std::vector<std::vector<std::complex<float>>> SignalService::GenerateFormCpu(
    const FormParams& params) {
    if (!form_gen_)
        form_gen_.emplace(backend_);
    form_gen_->SetParams(params);
    return form_gen_->GenerateToCpu();
}

// ════════════════════════════════════════════════════════════════════════════
// DelayedFormSignal generation (Farrow 48×5)
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem> SignalService::GenerateDelayedFormGpu(
    const FormParams& params, const std::vector<float>& delay_us) {
    if (!delayed_gen_)
        delayed_gen_.emplace(backend_);
    delayed_gen_->SetParams(params);
    delayed_gen_->SetDelays(delay_us);
    return delayed_gen_->GenerateInputData();
}

std::vector<std::vector<std::complex<float>>> SignalService::GenerateDelayedFormCpu(
    const FormParams& params, const std::vector<float>& delay_us) {
    if (!delayed_gen_)
        delayed_gen_.emplace(backend_);
    delayed_gen_->SetParams(params);
    delayed_gen_->SetDelays(delay_us);
    return delayed_gen_->GenerateToCpu();
}

} // namespace signal_gen

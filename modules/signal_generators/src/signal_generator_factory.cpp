/**
 * @file signal_generator_factory.cpp
 * @brief Реализация фабрики генераторов
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "signal_generator_factory.hpp"
#include "generators/cw_generator.hpp"
#include "generators/lfm_generator.hpp"
#include "generators/noise_generator.hpp"

#include <stdexcept>

namespace signal_gen {

std::unique_ptr<ISignalGenerator> SignalGeneratorFactory::CreateCw(
    drv_gpu_lib::IBackend* backend, const CwParams& params) {
    return std::make_unique<CwGenerator>(backend, params);
}

std::unique_ptr<ISignalGenerator> SignalGeneratorFactory::CreateLfm(
    drv_gpu_lib::IBackend* backend, const LfmParams& params) {
    return std::make_unique<LfmGenerator>(backend, params);
}

std::unique_ptr<ISignalGenerator> SignalGeneratorFactory::CreateNoise(
    drv_gpu_lib::IBackend* backend, const NoiseParams& params) {
    return std::make_unique<NoiseGenerator>(backend, params);
}

std::unique_ptr<ISignalGenerator> SignalGeneratorFactory::Create(
    drv_gpu_lib::IBackend* backend, const SignalRequest& request)
{
    switch (request.kind) {
        case SignalKind::CW:
            return CreateCw(backend, std::get<CwParams>(request.params));
        case SignalKind::LFM:
            return CreateLfm(backend, std::get<LfmParams>(request.params));
        case SignalKind::NOISE:
            return CreateNoise(backend, std::get<NoiseParams>(request.params));
        default:
            throw std::invalid_argument("Unknown SignalKind");
    }
}

} // namespace signal_gen

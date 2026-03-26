#pragma once

/**
 * @file signal_generator_factory.hpp
 * @brief Фабрика генераторов сигналов (Factory pattern)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "i_signal_generator.hpp"
#include "params/signal_request.hpp"
#if !ENABLE_ROCM
#include "generators/form_signal_generator.hpp"
#endif
#if ENABLE_ROCM
#include "generators/form_signal_generator_rocm.hpp"
#endif
#include "generators/form_script_generator.hpp"
#include "interface/i_backend.hpp"
#include "common/backend_type.hpp"

#include <memory>

namespace signal_gen {

/**
 * @class SignalGeneratorFactory
 * @brief Создаёт генераторы по типу сигнала
 */
class SignalGeneratorFactory {
public:
    /// Создать CW генератор
    static std::unique_ptr<ISignalGenerator> CreateCw(
        drv_gpu_lib::IBackend* backend, const CwParams& params);

    /// Создать LFM генератор
    static std::unique_ptr<ISignalGenerator> CreateLfm(
        drv_gpu_lib::IBackend* backend, const LfmParams& params);

    /// Создать Noise генератор
    static std::unique_ptr<ISignalGenerator> CreateNoise(
        drv_gpu_lib::IBackend* backend, const NoiseParams& params);

#if !ENABLE_ROCM
    /// Создать FormSignalGenerator (OpenCL, standalone, не ISignalGenerator)
    static std::unique_ptr<FormSignalGenerator> CreateForm(
        drv_gpu_lib::IBackend* backend, const FormParams& params);
#endif

#if ENABLE_ROCM
    /// Создать FormSignalGeneratorROCm (ROCm, standalone)
    static std::unique_ptr<FormSignalGeneratorROCm> CreateFormROCm(
        drv_gpu_lib::IBackend* backend, const FormParams& params);
#endif

    /// Создать FormScriptGenerator (DSL + on-disk cache)
    static std::unique_ptr<FormScriptGenerator> CreateFormScript(
        drv_gpu_lib::IBackend* backend, const FormParams& params);

    /// Создать генератор по SignalRequest
    static std::unique_ptr<ISignalGenerator> Create(
        drv_gpu_lib::IBackend* backend, const SignalRequest& request);
};

} // namespace signal_gen

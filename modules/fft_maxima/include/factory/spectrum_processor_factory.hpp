#pragma once

/**
 * @file spectrum_processor_factory.hpp
 * @brief Factory for creating ISpectrumProcessor by DriverType
 *
 * GRASP: Creator — factory creates processor instances.
 * Part of SpectrumMaximaFinder refactoring (Phase 2).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "interface/i_spectrum_processor.hpp"
#include "interface/spectrum_input_data.hpp"
#include "interface/i_backend.hpp"

#include <memory>

namespace antenna_fft {

/**
 * @class SpectrumProcessorFactory
 * @brief Creates ISpectrumProcessor by DriverType
 *
 * Usage:
 *   auto proc = SpectrumProcessorFactory::Create(DriverType::OPENCL, backend);
 *   auto proc = SpectrumProcessorFactory::Create(DriverType::ROCM, backend);  // stub
 */
class SpectrumProcessorFactory {
public:
    /**
     * @brief Create processor for given driver type
     * @param driver OPENCL or ROCM (AUTO defaults to OPENCL)
     * @param backend DrvGPU backend (non-owning)
     * @return unique_ptr to processor, never null
     * @throws std::runtime_error if ROCm requested (not implemented)
     */
    static std::unique_ptr<ISpectrumProcessor> Create(
        DriverType driver,
        drv_gpu_lib::IBackend* backend);
};

} // namespace antenna_fft

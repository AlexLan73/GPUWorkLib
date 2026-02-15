/**
 * @file spectrum_processor_factory.cpp
 * @brief Factory implementation for ISpectrumProcessor
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "factory/spectrum_processor_factory.hpp"
#include "processors/spectrum_processor_opencl.hpp"
#include "processors/spectrum_processor_rocm.hpp"

#include <stdexcept>

namespace antenna_fft {

std::unique_ptr<ISpectrumProcessor> SpectrumProcessorFactory::Create(
    DriverType driver,
    drv_gpu_lib::IBackend* backend)
{
    if (!backend) {
        throw std::invalid_argument("SpectrumProcessorFactory: backend cannot be null");
    }

    DriverType effective = (driver == DriverType::AUTO) ? DriverType::OPENCL : driver;

    switch (effective) {
    case DriverType::OPENCL:
        return std::make_unique<SpectrumProcessorOpenCL>(backend);
    case DriverType::ROCM:
        return std::make_unique<SpectrumProcessorROCm>(backend);
    case DriverType::AUTO:
        break;  // already handled
    }
    throw std::invalid_argument("SpectrumProcessorFactory: unknown DriverType");
}

} // namespace antenna_fft

/**
 * @file spectrum_processor_factory.cpp
 * @brief Factory implementation for ISpectrumProcessor
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "factory/spectrum_processor_factory.hpp"
#include "processors/spectrum_processor_opencl.hpp"
#if ENABLE_ROCM
#include "processors/spectrum_processor_rocm.hpp"
#endif

#include <stdexcept>

namespace antenna_fft {

std::unique_ptr<ISpectrumProcessor> SpectrumProcessorFactory::Create(
    drv_gpu_lib::BackendType backend_type,
    drv_gpu_lib::IBackend* backend)
{
    if (!backend) {
        throw std::invalid_argument("SpectrumProcessorFactory: backend cannot be null");
    }

    auto effective = (backend_type == drv_gpu_lib::BackendType::AUTO)
        ? drv_gpu_lib::BackendType::OPENCL : backend_type;

    switch (effective) {
    case drv_gpu_lib::BackendType::OPENCL:
        return std::make_unique<SpectrumProcessorOpenCL>(backend);
    case drv_gpu_lib::BackendType::ROCm:
#if ENABLE_ROCM
        return std::make_unique<SpectrumProcessorROCm>(backend);
#else
        throw std::runtime_error("SpectrumProcessorFactory: ROCm backend not available (ENABLE_ROCM=0)");
#endif
    case drv_gpu_lib::BackendType::AUTO:
    case drv_gpu_lib::BackendType::OPENCLandROCm:
        return std::make_unique<SpectrumProcessorOpenCL>(backend);
    }
    throw std::invalid_argument("SpectrumProcessorFactory: unknown BackendType");
}

} // namespace antenna_fft

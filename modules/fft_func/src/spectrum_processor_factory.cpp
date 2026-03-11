/**
 * @file spectrum_processor_factory.cpp
 * @brief Factory implementation for ISpectrumProcessor
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "factory/spectrum_processor_factory.hpp"
#if ENABLE_CLFFT
#include "processors/spectrum_processor_opencl.hpp"
#endif
#if ENABLE_ROCM
#include "processors/spectrum_processor_rocm.hpp"
#endif

#include <stdexcept>

namespace antenna_fft {

/**
 * @brief Создать ISpectrumProcessor для указанного backend'а
 *
 * AUTO → ROCm (если доступен) или OPENCL (при ENABLE_CLFFT=1).
 * ROCm доступен только при ENABLE_ROCM=1 (Linux + AMD GPU).
 * OPENCL доступен только при ENABLE_CLFFT=1 (NVIDIA + clFFT, ветка opencl-clfft).
 * OPENCLandROCm → ROCm (если доступен), иначе OPENCL.
 *
 * @param backend_type OPENCL, ROCm, AUTO или OPENCLandROCm
 * @param backend      Указатель на IBackend (не владеет)
 * @return unique_ptr<ISpectrumProcessor>, никогда nullptr
 * @throws std::invalid_argument если backend == nullptr
 * @throws std::runtime_error    если запрошенный бэкенд недоступен
 */
std::unique_ptr<ISpectrumProcessor> SpectrumProcessorFactory::Create(
    drv_gpu_lib::BackendType backend_type,
    drv_gpu_lib::IBackend* backend)
{
    if (!backend) {
        throw std::invalid_argument("SpectrumProcessorFactory: backend cannot be null");
    }

    // AUTO: предпочитаем ROCm (если доступен), иначе OpenCL
    auto effective = backend_type;
    if (effective == drv_gpu_lib::BackendType::AUTO ||
        effective == drv_gpu_lib::BackendType::OPENCLandROCm)
    {
#if ENABLE_ROCM
        effective = drv_gpu_lib::BackendType::ROCm;
#elif ENABLE_CLFFT
        effective = drv_gpu_lib::BackendType::OPENCL;
#else
        throw std::runtime_error("SpectrumProcessorFactory: no backend available (ENABLE_ROCM=0, ENABLE_CLFFT=0)");
#endif
    }

    switch (effective) {
    case drv_gpu_lib::BackendType::OPENCL:
#if ENABLE_CLFFT
        return std::make_unique<SpectrumProcessorOpenCL>(backend);
#else
        throw std::runtime_error("SpectrumProcessorFactory: OpenCL/clFFT backend not available (ENABLE_CLFFT=0)");
#endif
    case drv_gpu_lib::BackendType::ROCm:
#if ENABLE_ROCM
        return std::make_unique<SpectrumProcessorROCm>(backend);
#else
        throw std::runtime_error("SpectrumProcessorFactory: ROCm backend not available (ENABLE_ROCM=0)");
#endif
    default:
        break;
    }
    throw std::invalid_argument("SpectrumProcessorFactory: unknown BackendType");
}

} // namespace antenna_fft

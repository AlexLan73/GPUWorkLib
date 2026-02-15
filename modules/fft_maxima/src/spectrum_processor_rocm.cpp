/**
 * @file spectrum_processor_rocm.cpp
 * @brief ROCm stub — all methods throw "not implemented"
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "processors/spectrum_processor_rocm.hpp"

#include <stdexcept>

namespace antenna_fft {

SpectrumProcessorROCm::SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {}

void SpectrumProcessorROCm::Initialize(const SpectrumParams&) {
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>&)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromGPU(
    void*, size_t, size_t, size_t)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatch(
    const std::vector<std::complex<float>>&, size_t, size_t)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatchFromGPU(
    void*, size_t, size_t, size_t)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>&,
    OutputDestination, uint32_t, uint32_t)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromGPUPipeline(
    void*, size_t, size_t, size_t,
    OutputDestination, uint32_t, uint32_t)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

AllMaximaResult SpectrumProcessorROCm::AllMaximaFromCPU(
    const std::vector<std::complex<float>>&,
    uint32_t, uint32_t, float,
    OutputDestination, uint32_t, uint32_t)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

AllMaximaResult SpectrumProcessorROCm::FindAllMaxima(
    void*, uint32_t, uint32_t, float,
    OutputDestination, uint32_t, uint32_t)
{
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

ProfilingData SpectrumProcessorROCm::GetProfilingData() const {
    return {};
}

void SpectrumProcessorROCm::ReallocateBuffersForBatch(size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

size_t SpectrumProcessorROCm::CalculateBytesPerAntenna() const {
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

void SpectrumProcessorROCm::CompilePostKernel() {
    throw std::runtime_error("SpectrumProcessorROCm: not implemented");
}

} // namespace antenna_fft

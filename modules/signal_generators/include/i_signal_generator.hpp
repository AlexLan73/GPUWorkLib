#pragma once

/**
 * @file i_signal_generator.hpp
 * @brief Интерфейс генератора сигналов (Strategy pattern)
 *
 * Каждый генератор (CW, LFM, Noise) реализует этот интерфейс.
 * Поддерживает генерацию на CPU (reference) и GPU (production).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "params/signal_request.hpp"
#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cstddef>

namespace signal_gen {

/**
 * @class ISignalGenerator
 * @brief Абстрактный интерфейс генератора сигналов
 *
 * Реализации:
 * - CwGenerator  — синусоида (CW)
 * - LfmGenerator — линейная ЧМ (chirp)
 * - NoiseGenerator — шум (Gaussian / White)
 */
class ISignalGenerator {
public:
    virtual ~ISignalGenerator() = default;

    /**
     * @brief Генерация на CPU (reference implementation)
     *
     * @param system  Параметры дискретизации (fs, length)
     * @param out     Выходной буфер [length] complex<float>
     * @param out_size Размер выходного буфера (должен >= length)
     */
    virtual void GenerateToCpu(
        const SystemSampling& system,
        std::complex<float>* out,
        size_t out_size) = 0;

    /**
     * @brief Генерация на GPU (production)
     *
     * @param system      Параметры дискретизации
     * @param beam_count  Количество лучей (1 = один сигнал)
     * @return cl_mem буфер [beam_count * length * sizeof(complex<float>)]
     *
     * @note Вызывающий код должен освободить cl_mem через clReleaseMemObject()!
     */
    virtual cl_mem GenerateToGpu(
        const SystemSampling& system,
        size_t beam_count = 1) = 0;

    /// Тип сигнала
    virtual SignalKind Kind() const = 0;
};

} // namespace signal_gen

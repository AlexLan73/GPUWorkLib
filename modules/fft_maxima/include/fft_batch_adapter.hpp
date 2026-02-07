#pragma once

/**
 * @file fft_batch_adapter.hpp
 * @brief Адаптер для использования DrvGPU::BatchManager в fft_maxima
 *
 * ============================================================================
 * ПРОБЛЕМА (пункт 5 плана):
 *   AntennaFFTCore содержит собственную BatchConfig с hard-coded параметрами.
 *   BatchManager из DrvGPU/services/ — более продвинутый:
 *   - Запрашивает реальную память GPU
 *   - Smart tail merging (хвосты < min_tail объединяются с предыдущим)
 *   - Универсален для любого модуля
 *
 * РЕШЕНИЕ:
 *   FFTBatchAdapter — тонкий адаптер, который:
 *   1. Рассчитывает per-item memory для FFT
 *   2. Использует BatchManager для оптимального разбиения
 *   3. Возвращает BatchRange[], совместимый с AntennaFFTCore::ProcessBatch()
 *
 * ИСПОЛЬЗОВАНИЕ:
 *   FFTBatchAdapter adapter(params, nFFT);
 *
 *   auto batches = adapter.CalculateBatches(backend);
 *
 *   for (auto& batch : batches) {
 *       auto results = fft_processor.ProcessBatch(
 *           input, batch.start, batch.count, &profiling);
 *   }
 *
 * ПРИМЕЧАНИЕ:
 *   Этот файл НЕ заменяет встроенную batch-логику в AntennaFFTCore.
 *   Он предлагает альтернативный путь для тех, кто хочет
 *   использовать DrvGPU::BatchManager напрямую.
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "interface/antenna_fft_params.h"
#include "DrvGPU/services/batch_manager.hpp"
#include "DrvGPU/common/i_backend.hpp"

#include <complex>
#include <cstddef>
#include <iostream>

namespace antenna_fft {

// ============================================================================
// FFTBatchAdapter — связка между AntennaFFT и DrvGPU::BatchManager
// ============================================================================

/**
 * @class FFTBatchAdapter
 * @brief Адаптер для расчёта FFT batch'ей через DrvGPU::BatchManager
 *
 * Знает формулу расчёта памяти на один beam (item):
 *   per_beam_memory = 2 * nFFT * sizeof(complex<float>)   // input + output FFT
 *                   + 2 * out_count_points_fft * sizeof(complex<float>)  // selected
 *                   + max_peaks_count * 32                 // maxima structs
 *                   + callback_overhead                    // pre/post callback userdata
 *
 * И передаёт эту информацию в BatchManager для оптимального разбиения.
 */
class FFTBatchAdapter {
public:
    /**
     * @brief Конструктор
     * @param params FFT параметры (beam_count, count_points, etc.)
     * @param nFFT   Вычисленный размер FFT (степень двойки)
     */
    FFTBatchAdapter(const AntennaFFTParams& params, size_t nFFT)
        : params_(params), nFFT_(nFFT) {
        CalculatePerBeamMemory();
    }

    /**
     * @brief Рассчитать оптимальные batch'и через BatchManager
     *
     * @param backend    Указатель на IBackend (для запроса GPU памяти)
     * @param min_tail   Минимальный хвост (если < min_tail, объединить с предыдущим)
     * @param mem_limit  Доля доступной памяти для использования (0.0 - 1.0)
     * @return Вектор BatchRange для обработки
     */
    std::vector<drv_gpu_lib::BatchRange> CalculateBatches(
        drv_gpu_lib::IBackend* backend,
        size_t min_tail = 3,
        double mem_limit = 0.7) const
    {
        // Оптимальный размер batch через BatchManager
        size_t batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
            backend,
            params_.beam_count,
            per_beam_bytes_,
            mem_limit
        );

        // Создаём batch'и с smart tail merging
        auto batches = drv_gpu_lib::BatchManager::CreateBatches(
            params_.beam_count,
            batch_size,
            min_tail,
            true  // merge_small_tail = true
        );

        return batches;
    }

    /**
     * @brief Проверить, нужно ли батчирование
     *
     * @param backend   Указатель на IBackend
     * @param mem_limit Доля доступной памяти
     * @return true если все beams помещаются в память без батчирования
     */
    bool AllBeamsFit(drv_gpu_lib::IBackend* backend, double mem_limit = 0.7) const {
        return drv_gpu_lib::BatchManager::AllItemsFit(
            backend, params_.beam_count, per_beam_bytes_, mem_limit);
    }

    /**
     * @brief Получить рассчитанную память на один beam (в байтах)
     */
    size_t GetPerBeamMemory() const { return per_beam_bytes_; }

    /**
     * @brief Получить общую требуемую память для всех beams
     */
    size_t GetTotalRequiredMemory() const {
        return per_beam_bytes_ * params_.beam_count;
    }

    /**
     * @brief Вывести информацию о расчёте памяти
     */
    void PrintMemoryInfo() const {
        std::cout << "  FFTBatchAdapter Memory Calculation:\n";
        std::cout << "    nFFT = " << nFFT_ << "\n";
        std::cout << "    Per beam:\n";
        std::cout << "      FFT buffers:    " << fft_buffer_bytes_ << " bytes\n";
        std::cout << "      Selected:       " << selected_bytes_ << " bytes\n";
        std::cout << "      Maxima:         " << maxima_bytes_ << " bytes\n";
        std::cout << "      Callback data:  " << callback_bytes_ << " bytes\n";
        std::cout << "      TOTAL per beam: " << per_beam_bytes_ << " bytes ("
                  << (per_beam_bytes_ / 1024.0) << " KB)\n";
        std::cout << "    Total for all " << params_.beam_count << " beams: "
                  << (GetTotalRequiredMemory() / (1024.0 * 1024.0)) << " MB\n\n";
    }

private:
    /**
     * @brief Рассчитать потребление памяти на один beam
     *
     * Формула повторяет AllocateBuffers() из AntennaFFTProcMax:
     *   - buffer_fft_input_:  nFFT * sizeof(complex<float>)
     *   - buffer_fft_output_: nFFT * sizeof(complex<float>)
     *   - buffer_selected_complex_: out_count_points_fft * sizeof(complex<float>)
     *   - buffer_selected_magnitude_: out_count_points_fft * sizeof(float)
     *   - buffer_maxima_: max_peaks_count * 32  (MaxValue struct = 32 bytes)
     *   - pre_callback_userdata_: 32 + count_points * sizeof(complex<float>)
     *   - post_callback_userdata_: 16 + out_count_points_fft * (sizeof(complex<float>) + sizeof(float))
     */
    void CalculatePerBeamMemory() {
        // FFT input + output буферы
        fft_buffer_bytes_ = 2 * nFFT_ * sizeof(std::complex<float>);

        // Selected spectrum (complex + magnitude)
        selected_bytes_ = params_.out_count_points_fft * sizeof(std::complex<float>)
                        + params_.out_count_points_fft * sizeof(float);

        // Maxima buffer (MaxValue struct = 32 bytes)
        maxima_bytes_ = params_.max_peaks_count * 32;

        // Callback userdata (header + input data per beam)
        // pre-callback: 32 bytes header + count_points * complex<float>
        // post-callback: 16 bytes header + out_count_points * (complex<float> + float)
        callback_bytes_ = 32 + params_.count_points * sizeof(std::complex<float>)
                        + 16 + params_.out_count_points_fft * (sizeof(std::complex<float>) + sizeof(float));

        // Итого на один beam
        per_beam_bytes_ = fft_buffer_bytes_ + selected_bytes_ + maxima_bytes_ + callback_bytes_;
    }

    // Параметры
    AntennaFFTParams params_;
    size_t nFFT_;

    // Рассчитанные размеры
    size_t fft_buffer_bytes_ = 0;
    size_t selected_bytes_ = 0;
    size_t maxima_bytes_ = 0;
    size_t callback_bytes_ = 0;
    size_t per_beam_bytes_ = 0;
};

} // namespace antenna_fft

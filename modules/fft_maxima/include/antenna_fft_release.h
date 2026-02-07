#pragma once

/**
 * @file antenna_fft_release.h
 * @brief Release-реализация FFT с колбэками clFFT
 *
 * Высокопроизводительная реализация с pre/post колбэками clFFT
 * для zero-copy обработки на GPU.
 *
 * Конвейер: pre-callback (дополнение) -> FFT -> post-callback (амплитуда + выбор)
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include "antenna_fft_core.h"
#include "kernels/fft_kernel_sources.hpp"
#include "fft_plan_cache.hpp"

#include <memory>

namespace antenna_fft {

/**
 * @class AntennaFFTProcMax
 * @brief Release-реализация — колбэки clFFT для максимальной производительности
 *
 * Продакшен-класс для FFT-обработки.
 * Вся обработка выполняется одним вызовом clFFT с колбэками.
 *
 * Конвейер:
 * 1. Pre-callback: чтение входных данных + дополнение до nFFT
 * 2. clFFT: прямое FFT
 * 3. Post-callback: fftshift + расчёт амплитуды + выбор out_count_points_fft
 *
 * Использование:
 * ```cpp
 * AntennaFFTProcMax fft(params, backend);
 * auto result = fft.ProcessNew(input_data);
 * ```
 */
class AntennaFFTProcMax : public AntennaFFTCore {
public:
    /**
     * @brief Конструктор
     * @param params Параметры обработки
     * @param backend Указатель на IBackend (поддержка Multi-GPU)
     */
    explicit AntennaFFTProcMax(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend);

    /**
     * @brief Деструктор
     */
    ~AntennaFFTProcMax() override;

    // Запрет копирования, разрешено перемещение
    AntennaFFTProcMax(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax& operator=(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax(AntennaFFTProcMax&&) noexcept = default;
    AntennaFFTProcMax& operator=(AntennaFFTProcMax&&) noexcept = default;

protected:
    // ═══════════════════════════════════════════════════════════════════════════
    // Реализации виртуальных методов
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать FFT-план с колбэками
     */
    void Initialize() override;

    /**
     * @brief Обработать все лучи одним пакетом (через колбэки)
     */
    AntennaFFTResult ProcessSingleBatch(cl_mem input_signal) override;

    /**
     * @brief Обработать один пакет (через колбэки)
     */
    std::vector<FFTResult> ProcessBatch(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        BatchProfilingData* out_profiling = nullptr) override;

    /**
     * @brief Выделить буферы для обработки с колбэками
     */
    void AllocateBuffers(size_t num_beams) override;

    /**
     * @brief Освободить выделенные буферы
     */
    void ReleaseBuffers() override;

private:
    // ═══════════════════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Создать FFT-план с pre- и post-колбэками
     * @param num_beams Количество лучей (размер пакета)
     */
    void CreateFFTPlanWithCallbacks(size_t num_beams);

    /**
     * @brief Выполнить FFT с колбэками
     * @param input_signal Буфер входных данных
     * @param num_beams Количество лучей для обработки
     * @param start_beam Начальный индекс луча (для батчинга)
     * @param out_fft_event Событие завершения FFT
     * @return true при успехе
     */
    bool ExecuteFFTWithCallbacks(
        cl_mem input_signal,
        size_t num_beams,
        size_t start_beam,
        cl_event* out_fft_event);

    /**
     * @brief Прочитать результаты с GPU после FFT
     * @param num_beams Количество лучей
     * @param start_beam Начальный индекс луча
     * @return Результаты по обработанным лучам
     */
    std::vector<FFTResult> ReadResults(size_t num_beams, size_t start_beam);

    // ═══════════════════════════════════════════════════════════════════════════
    // Приватные поля
    // ═══════════════════════════════════════════════════════════════════════════

    // Буферы выбранного спектра (результат post-callback)
    cl_mem buffer_selected_complex_;       // Комплексные значения выбранных точек
    cl_mem buffer_selected_magnitude_;     // Магнитуды выбранных точек

    // Параметры закешированного плана
    size_t plan_num_beams_;                // Количество лучей, для которого создан план

    // Кэш FFT-планов (избегаем дорогого пересоздания)
    std::unique_ptr<FFTPlanCache> plan_cache_;
};

} // namespace antenna_fft

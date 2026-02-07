#pragma once

/**
 * @file antenna_fft_core.h
 * @brief Базовый абстрактный класс для FFT с поиском максимумов
 *
 * Общая функциональность для Release и Debug реализаций.
 * Содержит общую логику пакетной обработки, управление буферами и утилиты.
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include "interface/antenna_fft_params.h"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <clFFT.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <chrono>

namespace antenna_fft {

/**
 * @class AntennaFFTCore
 * @brief Абстрактный базовый класс для FFT-обработки
 *
 * Предоставляет общую функциональность:
 * - Логика пакетной обработки (ProcessWithBatching)
 * - Выделение буферов
 * - Управление FFT-планом
 * - Профилирование
 *
 * Производные классы реализуют:
 * - ProcessSingleBatch() — обработка одного пакета
 * - Initialize() — специфичная инициализация (колбэки или ядра)
 */
class AntennaFFTCore {
public:
    // ═══════════════════════════════════════════════════════════════════════════
    // Публичные типы
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Данные профилирования для одного пакета
     */
    struct BatchProfilingData {
        size_t batch_index = 0;
        size_t start_beam = 0;
        size_t num_beams = 0;
        double padding_time_ms = 0.0;
        double fft_time_ms = 0.0;
        double post_time_ms = 0.0;
        double gpu_time_ms = 0.0;
    };

    /**
     * @brief Конфигурация пакетной обработки
     */
    struct BatchConfig {
        double memory_usage_limit = 0.65;    // 65% доступной памяти
        double batch_size_ratio = 0.22;      // 22% лучей на пакет
        size_t min_beams_for_batch = 10;     // Минимум лучей для пакетного режима
        size_t beams_per_batch = 0;          // Вычисленное число лучей на пакет
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Конструктор / Деструктор
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Конструктор с IBackend для поддержки Multi-GPU
     * @param params Параметры обработки
     * @param backend Указатель на IBackend (не синглтон!)
     */
    explicit AntennaFFTCore(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend);

    /**
     * @brief Виртуальный деструктор
     */
    virtual ~AntennaFFTCore();

    // Запрет копирования, разрешено перемещение
    AntennaFFTCore(const AntennaFFTCore&) = delete;
    AntennaFFTCore& operator=(const AntennaFFTCore&) = delete;
    AntennaFFTCore(AntennaFFTCore&&) noexcept;
    AntennaFFTCore& operator=(AntennaFFTCore&&) noexcept;

    // ═══════════════════════════════════════════════════════════════════════════
    // Публичный интерфейс (общий для всех реализаций)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Основная точка входа — обработка данных с CPU
     * @param input_data Вектор комплексных чисел (beam_count * count_points)
     * @return Результаты обработки по всем лучам
     */
    AntennaFFTResult ProcessNew(const std::vector<std::complex<float>>& input_data);

    /**
     * @brief Основная точка входа — обработка данных с GPU
     * @param input_signal Буфер GPU с входными данными
     * @return Результаты обработки по всем лучам
     */
    AntennaFFTResult ProcessNew(cl_mem input_signal);

    /**
     * @brief Обработка с разбиением на пакеты (общий цикл, виртуальный ProcessBatch)
     * @param input_signal Буфер GPU со всеми входными данными
     * @return Результаты обработки по всем лучам
     */
    AntennaFFTResult ProcessWithBatching(cl_mem input_signal);

    /**
     * @brief Получить последние результаты профилирования
     */
    const FFTProfilingResults& GetLastProfilingResults() const { return last_profiling_results_; }

    /**
     * @brief Получить вычисленный размер nFFT
     */
    size_t GetNFFT() const { return nFFT_; }

    /**
     * @brief Получить параметры обработки
     */
    const AntennaFFTParams& GetParams() const { return params_; }

    /**
     * @brief Получить данные профилирования по пакетам
     */
    const std::vector<BatchProfilingData>& GetBatchProfiling() const { return batch_profiling_; }

    /**
     * @brief Проверить, использовался ли пакетный режим в последней обработке
     */
    bool WasBatchModeUsed() const { return last_used_batch_mode_; }

protected:
    // ═══════════════════════════════════════════════════════════════════════════
    // Виртуальные методы (реализуются производными классами)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать специфичные ресурсы (FFT-планы, ядра)
     * Вызывается из конструктора после базовой инициализации
     */
    virtual void Initialize() = 0;

    /**
     * @brief Обработать один пакет (все лучи помещаются в память)
     * @param input_signal Буфер GPU с входными данными
     * @return Результаты обработки
     */
    virtual AntennaFFTResult ProcessSingleBatch(cl_mem input_signal) = 0;

    /**
     * @brief Обработать один пакет в режиме батчинга
     * @param input_signal Полный входной буфер (все лучи)
     * @param start_beam Начальный индекс луча
     * @param num_beams Количество лучей в пакете
     * @param out_profiling Опциональный вывод профилирования
     * @return Результаты по лучам этого пакета
     */
    virtual std::vector<FFTResult> ProcessBatch(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        BatchProfilingData* out_profiling = nullptr) = 0;

    /**
     * @brief Выделить буферы GPU для обработки
     * @param num_beams Количество лучей, на которое выделять
     */
    virtual void AllocateBuffers(size_t num_beams) = 0;

    /**
     * @brief Освободить выделенные буферы
     */
    virtual void ReleaseBuffers() = 0;

    // ═══════════════════════════════════════════════════════════════════════════
    // Защищённые утилиты (общая реализация)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Вычислить nFFT по count_points
     */
    size_t CalculateNFFT(size_t count_points) const;

    /**
     * @brief Проверить, является ли число степенью двойки
     */
    bool IsPowerOf2(size_t n) const;

    /**
     * @brief Найти следующую большую степень двойки
     */
    size_t NextPowerOf2(size_t n) const;

    /**
     * @brief Оценить требуемую память для заданного числа лучей
     */
    size_t EstimateRequiredMemory(size_t num_beams) const;

    /**
     * @brief Проверить, достаточно ли доступной памяти
     */
    bool CheckAvailableMemory(size_t required_memory, double threshold = 0.4) const;

    /**
     * @brief Вычислить конфигурацию пакетов
     */
    void CalculateBatchConfig();

    /**
     * @brief Проверить, нужна ли пакетная обработка
     */
    bool NeedsBatching() const;

    /**
     * @brief Создать входной буфер из данных CPU
     */
    cl_mem CreateInputBuffer(const std::vector<std::complex<float>>& input_data);

    /**
     * @brief Создать буфер userdata для pre-callback
     */
    void CreatePreCallbackUserData(size_t num_beams);

    /**
     * @brief Создать буфер userdata для post-callback
     */
    void CreatePostCallbackUserData(size_t num_beams);

    /**
     * @brief Профилировать событие OpenCL
     */
    double ProfileEvent(cl_event event, const std::string& operation_name);

    /**
     * @brief Освободить FFT-план
     */
    void ReleaseFFTPlan();

    // ═══════════════════════════════════════════════════════════════════════════
    // Защищённые поля (доступны производным классам)
    // ═══════════════════════════════════════════════════════════════════════════

    AntennaFFTParams params_;              // Параметры обработки
    size_t nFFT_;                          // Вычисленный размер FFT

    // Бэкенд DrvGPU (поддержка Multi-GPU)
    drv_gpu_lib::IBackend* backend_;       // Указатель на бэкенд (не владеет)

    // Ресурсы OpenCL (получаются из бэкенда)
    cl_context context_;                   // Контекст OpenCL
    cl_command_queue queue_;               // Очередь команд
    cl_device_id device_;                  // Устройство OpenCL

    // Ресурсы clFFT
    clfftPlanHandle plan_handle_;          // Хэндл FFT-плана
    bool plan_created_;                    // Флаг создания плана

    // Общие буферы GPU
    cl_mem buffer_fft_input_;              // Входной буфер FFT (nFFT * beam_count)
    cl_mem buffer_fft_output_;             // Выходной буфер FFT
    cl_mem buffer_maxima_;                 // Буфер максимумов

    // Буферы userdata для колбэков
    cl_mem pre_callback_userdata_;         // Userdata для pre-callback
    cl_mem post_callback_userdata_;        // Userdata для post-callback

    // Профилирование
    FFTProfilingResults last_profiling_results_;
    std::vector<BatchProfilingData> batch_profiling_;
    double batch_total_cpu_time_ms_;
    bool last_used_batch_mode_;

    // Конфигурация пакетов
    BatchConfig batch_config_;
    size_t current_buffer_beams_;          // Текущий выделенный размер буфера
};

} // namespace antenna_fft

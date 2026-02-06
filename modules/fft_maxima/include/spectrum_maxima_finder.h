#pragma once

/**
 * @file spectrum_maxima_finder.h
 * @brief Класс для поиска максимума спектра FFT с параболической интерполяцией
 *
 * Реализует:
 * - Pre-callback для padding и repeat_count
 * - Post-kernel для поиска максимума и параболической интерполяции
 * - Профилирование средствами GPU
 * - Работа через DrvGPU с SVM
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-06
 */

#include "common/i_backend.hpp"
#include "kernels/fft_kernel_sources.hpp"

#include <CL/cl.h>
#include <clFFT.h>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Структуры данных
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct SpectrumParams
 * @brief Параметры для поиска максимума спектра
 */
struct SpectrumParams {
    uint32_t antenna_count = 5;         ///< Количество антен (1-256)
    uint32_t n_point = 1000;            ///< Точек на антену (исходный размер сигнала)
    uint32_t repeat_count = 2;          ///< Множитель размера FFT (2^n: 1,2,4,8...)
    float sample_rate = 1000.0f;        ///< Частота дискретизации (Гц)
    uint32_t search_range = 0;          ///< Диапазон поиска максимума (0 = авто = nFFT/4)

    // Вычисляемые параметры (заполняются в Initialize)
    uint32_t nFFT = 0;                  ///< Размер FFT = nextPow2(n_point) * repeat_count
    uint32_t base_fft = 0;              ///< Базовый размер = nextPow2(n_point)
};

/**
 * @struct MaxValue
 * @brief Результат поиска максимума (должен совпадать с GPU структурой!)
 */
struct MaxValue {
    uint32_t index;             ///< Индекс в FFT спектре
    float real;                 ///< Re компонента
    float imag;                 ///< Im компонента
    float magnitude;            ///< |magnitude| = sqrt(re^2 + im^2)
    float phase;                ///< Фаза в градусах
    float freq_offset;          ///< Параболическая поправка [-0.5, 0.5]
    float refined_frequency;    ///< Уточнённая частота (Гц)
    uint32_t pad;               ///< Padding для выравнивания (32 bytes total)
};

/**
 * @struct SpectrumResult
 * @brief Результат обработки для одной антены
 */
struct SpectrumResult {
    uint32_t antenna_id;        ///< Номер антены
    MaxValue interpolated;      ///< Результат параболической интерполяции
    MaxValue left_point;        ///< Левая точка (index-1)
    MaxValue center_point;      ///< Центральная точка (максимум)
    MaxValue right_point;       ///< Правая точка (index+1)
};

/**
 * @struct ProfilingData
 * @brief Данные профилирования GPU
 */
struct ProfilingData {
    double upload_time_ms = 0.0;        ///< Время загрузки данных Host→GPU
    double fft_time_ms = 0.0;           ///< Время выполнения FFT (с pre-callback)
    double post_kernel_time_ms = 0.0;   ///< Время выполнения post-kernel
    double download_time_ms = 0.0;      ///< Время выгрузки результатов GPU→Host
    double total_time_ms = 0.0;         ///< Общее время
};

// ════════════════════════════════════════════════════════════════════════════
// Класс SpectrumMaximaFinder
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class SpectrumMaximaFinder
 * @brief Поиск максимума спектра после FFT с параболической интерполяцией
 *
 * Алгоритм:
 * 1. Pre-callback: padding n_point → nFFT с нулями
 * 2. FFT: выполнение clFFT с встроенным pre-callback
 * 3. Post-kernel: поиск максимума + парабола (ОТДЕЛЬНЫЙ kernel)
 *
 * Почему post-kernel отдельный?
 * - Нужна редукция (поиск максимума среди всех точек)
 * - Использует __local memory и barrier()
 * - Невозможно реализовать как post-callback (он видит только 1 элемент)
 *
 * Использование:
 * @code
 * SpectrumParams params;
 * params.antenna_count = 5;
 * params.n_point = 1000;
 * params.repeat_count = 2;
 * params.sample_rate = 1000.0f;
 *
 * SpectrumMaximaFinder finder(params, backend);
 * finder.Initialize();
 *
 * auto results = finder.Process(input_data);
 * auto profiling = finder.GetProfilingData();
 * @endcode
 */
class SpectrumMaximaFinder {
public:
    // ═══════════════════════════════════════════════════════════════════════
    // Конструктор / Деструктор
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Конструктор
     * @param params Параметры обработки
     * @param backend Указатель на DrvGPU backend (не владеет)
     */
    explicit SpectrumMaximaFinder(const SpectrumParams& params,
                                   drv_gpu_lib::IBackend* backend);

    /**
     * @brief Деструктор (освобождает GPU ресурсы)
     */
    ~SpectrumMaximaFinder();

    // Запрет копирования
    SpectrumMaximaFinder(const SpectrumMaximaFinder&) = delete;
    SpectrumMaximaFinder& operator=(const SpectrumMaximaFinder&) = delete;

    // Разрешение перемещения
    SpectrumMaximaFinder(SpectrumMaximaFinder&&) noexcept;
    SpectrumMaximaFinder& operator=(SpectrumMaximaFinder&&) noexcept;

    // ═══════════════════════════════════════════════════════════════════════
    // Публичный интерфейс
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Инициализация GPU ресурсов
     *
     * Создаёт:
     * - Буферы GPU (pre_callback_userdata, fft_input/output, maxima)
     * - FFT план с pre-callback
     * - Компилирует post-kernel
     *
     * @throws std::runtime_error при ошибке инициализации
     */
    void Initialize();

    /**
     * @brief Обработка данных
     * @param input_data Входные данные [antenna_count × n_point] complex<float>
     * @return Вектор результатов для каждой антены
     * @throws std::runtime_error при ошибке обработки
     */
    std::vector<SpectrumResult> Process(
        const std::vector<std::complex<float>>& input_data);

    /**
     * @brief Получить данные профилирования последнего вызова
     */
    const ProfilingData& GetProfilingData() const { return profiling_; }

    /**
     * @brief Получить параметры (с вычисленными nFFT и т.д.)
     */
    const SpectrumParams& GetParams() const { return params_; }

    /**
     * @brief Проверить, инициализирован ли объект
     */
    bool IsInitialized() const { return initialized_; }

    /**
     * @brief Вывести информацию о конфигурации
     */
    void PrintInfo() const;

private:
    // ═══════════════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════════════

    /// Вычислить nFFT и другие параметры
    void CalculateFFTSize();

    /// Следующая степень двойки
    static uint32_t NextPowerOf2(uint32_t n);

    /// Создать GPU буферы
    void AllocateBuffers();

    /// Создать FFT план с pre-callback
    void CreateFFTPlanWithCallback();

    /// Скомпилировать post-kernel
    void CompilePostKernel();

    /// Загрузить данные в GPU
    cl_event UploadData(const std::vector<std::complex<float>>& input_data);

    /// Выполнить FFT
    cl_event ExecuteFFT(cl_event wait_event);

    /// Выполнить post-kernel
    cl_event ExecutePostKernel(cl_event wait_event);

    /// Прочитать результаты
    std::vector<SpectrumResult> ReadResults(cl_event wait_event);

    /// Профилирование события
    double ProfileEvent(cl_event event, const char* name);

    /// Освободить ресурсы
    void ReleaseResources();

    // ═══════════════════════════════════════════════════════════════════════
    // Приватные поля
    // ═══════════════════════════════════════════════════════════════════════

    // Параметры
    SpectrumParams params_;
    bool initialized_ = false;

    // DrvGPU backend
    drv_gpu_lib::IBackend* backend_ = nullptr;

    // OpenCL ресурсы
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;

    // clFFT
    clfftPlanHandle plan_handle_ = 0;
    bool plan_created_ = false;

    // GPU буферы
    cl_mem pre_callback_userdata_ = nullptr;    ///< [32 bytes params][input data]
    cl_mem fft_input_ = nullptr;                ///< FFT input buffer
    cl_mem fft_output_ = nullptr;               ///< FFT output buffer
    cl_mem maxima_output_ = nullptr;            ///< Post-kernel results

    // Post-kernel
    cl_program post_program_ = nullptr;
    cl_kernel post_kernel_ = nullptr;

    // Профилирование
    ProfilingData profiling_;

    // Константы
    static constexpr size_t PRE_CALLBACK_HEADER_SIZE = 32;  ///< Размер заголовка userdata
    static constexpr size_t LOCAL_SIZE = 256;               ///< Размер work-group для post-kernel
};

} // namespace antenna_fft

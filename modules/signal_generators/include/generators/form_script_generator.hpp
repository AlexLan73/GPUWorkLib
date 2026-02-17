#pragma once

/**
 * @file form_script_generator.hpp
 * @brief FormScriptGenerator — DSL-генератор сигналов с on-disk кэшем кернелов
 *
 * Генерирует OpenCL kernel из FormParams с поддержкой:
 * - DSL-представление (human-readable скрипт)
 * - Параметры как #define → оптимизация компилятором OpenCL
 * - On-disk кэш: сохранение .cl source + binary по имени
 * - Версионирование: коллизии → _00, _01, ...
 * - manifest.json: индекс кернелов с комментариями
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "../params/form_params.hpp"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace signal_gen {

/**
 * @struct KernelManifestEntry
 * @brief Запись в manifest.json об одном кернеле
 */
struct KernelManifestEntry {
  std::string name;           ///< Имя кернела (без расширения)
  std::string comment;        ///< Пользовательский комментарий
  std::string created;        ///< Дата создания ISO 8601
  std::string params_string;  ///< Строка параметров для восстановления
  std::string backend;        ///< "opencl" или "rocm"
};

/**
 * @class FormScriptGenerator
 * @brief DSL-генератор + on-disk кэш кернелов для формулы getX
 *
 * Два режима работы:
 * 1. **Из параметров**: SetParams() → Compile() → Generate()
 * 2. **Из кэша**: LoadKernel(name) → Generate()
 *
 * @code
 * FormScriptGenerator gen(backend);
 *
 * // Режим 1: из параметров
 * FormParams p;
 * p.f0 = 1e6; p.antennas = 8; p.points = 4096;
 * gen.SetParams(p);
 * gen.Compile();
 * cl_mem buf = gen.Generate();
 * gen.SaveKernel("my_signal", "CW 1MHz 8ch");
 *
 * // Режим 2: из кэша
 * gen.LoadKernel("my_signal");
 * cl_mem buf2 = gen.Generate();
 *
 * // DSL текст (для просмотра)
 * std::cout << gen.GenerateScript() << std::endl;
 * @endcode
 */
class FormScriptGenerator {
public:
  explicit FormScriptGenerator(drv_gpu_lib::IBackend* backend);
  ~FormScriptGenerator();

  // No copy
  FormScriptGenerator(const FormScriptGenerator&) = delete;
  FormScriptGenerator& operator=(const FormScriptGenerator&) = delete;

  // Move
  FormScriptGenerator(FormScriptGenerator&& other) noexcept;
  FormScriptGenerator& operator=(FormScriptGenerator&& other) noexcept;

  // ══════════════════════════════════════════════════════════════════════
  // Параметры
  // ══════════════════════════════════════════════════════════════════════

  void SetParams(const FormParams& params);
  void SetParamsFromString(const std::string& params_str);
  const FormParams& GetParams() const { return params_; }

  // ══════════════════════════════════════════════════════════════════════
  // DSL / Kernel source
  // ══════════════════════════════════════════════════════════════════════

  /// Генерация читаемого DSL-скрипта из текущих params
  std::string GenerateScript() const;

  /// Генерация полного OpenCL kernel source (с PRNG, #define params)
  std::string GenerateKernelSource() const;

  /// Компиляция kernel из текущих params
  void Compile();

  // ══════════════════════════════════════════════════════════════════════
  // On-disk кэш кернелов
  // ══════════════════════════════════════════════════════════════════════

  /**
   * @brief Сохранить текущий kernel на диск
   * @param name Имя (без расширения): "work_sig0"
   * @param comment Комментарий (записывается в manifest)
   *
   * Создаёт: name.cl + bin/name_opencl.bin
   * При коллизии: старые файлы → name_00.cl, name_opencl_00.bin
   */
  void SaveKernel(const std::string& name, const std::string& comment = "");

  /**
   * @brief Загрузить kernel с диска по имени
   * @param name Имя кернела (без расширения)
   *
   * Приоритет: binary (fast) → source (compile + save binary)
   */
  void LoadKernel(const std::string& name);

  /// Список имён кернелов из manifest.json
  std::vector<std::string> ListKernels() const;

  // ══════════════════════════════════════════════════════════════════════
  // Генерация сигнала
  // ══════════════════════════════════════════════════════════════════════

  /**
   * @brief Генерация на GPU
   * @return cl_mem [antennas * points * sizeof(complex<float>)]
   * @note Вызывающий код должен освободить через clReleaseMemObject()
   */
  cl_mem Generate();

  /**
   * @brief Генерация с возвратом на CPU (по каналам)
   * @return vector[antenna_id][sample_id] complex<float>
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  // ══════════════════════════════════════════════════════════════════════
  // Info
  // ══════════════════════════════════════════════════════════════════════

  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }

  bool IsReady() const { return program_ != nullptr; }
  const std::string& GetCurrentKernelSource() const { return kernel_source_; }

  /// Путь к директории кернелов
  static std::string GetKernelsDir();
  /// Путь к директории бинарников
  static std::string GetKernelsBinDir();

private:
  void CompileSource(const std::string& source);
  void ReleaseGpuResources();

  // On-disk helpers
  void VersionOldFiles(const std::string& name) const;
  void WriteManifestEntry(const std::string& name,
                          const std::string& comment) const;
  std::vector<unsigned char> GetProgramBinary() const;
  void LoadFromBinary(const std::vector<unsigned char>& binary);
  void LoadFromSource(const std::string& source);

  static std::string GetTimestamp();
  std::string ParamsToString() const;

  drv_gpu_lib::IBackend* backend_ = nullptr;
  FormParams params_;
  std::string kernel_source_;
  std::string loaded_kernel_name_;  ///< имя загруженного с диска кернела

  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen

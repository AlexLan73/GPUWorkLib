#pragma once

/**
 * @file fm_correlator.hpp
 * @brief FMCorrelator facade — high-level API for FM correlation
 *
 * Delegates GPU work to FMCorrelatorProcessorROCm.
 * Provides M-sequence generation (CPU LFSR) and two PrepareReference modes:
 *   - External ref vector
 *   - Internal M-seq generation from params_.lfsr_seed
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

#include "fm_correlator_types.hpp"
#include "fm_correlator_processor_rocm.hpp"
#include "interface/i_backend.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace drv_gpu_lib {

// Фасад FM-коррелятора: публичный API поверх FMCorrelatorProcessorROCm.
// Зачем фасад: изолирует пользователя от hipFFT/hiprtc деталей, предоставляет
// удобные хелперы (GenerateMSequence, auto-batching).
// Порядок вызовов: SetParams() → PrepareReference() → Process() / RunTestPattern().
class FMCorrelator {
public:
  // backend — не владеем, backend должен жить дольше FMCorrelator.
  explicit FMCorrelator(IBackend* backend);
  ~FMCorrelator() = default;

  FMCorrelator(const FMCorrelator&) = delete;
  FMCorrelator& operator=(const FMCorrelator&) = delete;

  // Применить новые параметры: пересоздаёт GPU-буферы, FFT-планы, сбрасывает
  // ref_prepared_ — после SetParams() обязателен повторный PrepareReference().
  void SetParams(const FMCorrelatorParams& params);

  // CPU-генерация M-последовательности через LFSR (использует params_.lfsr_seed).
  // Caller owns возвращённый вектор. Можно передать результат в PrepareReference().
  std::vector<float> GenerateMSequence() const;

  // Overload с явным seed — когда нужна последовательность с другим начальным
  // состоянием (например, для сравнения или Python-биндинга).
  std::vector<float> GenerateMSequence(uint32_t seed) const;

  // Загрузить готовый эталонный сигнал: H2D → apply_cyclic_shifts → C2C FFT.
  // ref.size() должен == params_.fft_size.
  void PrepareReference(const std::vector<float>& ref);

  // Сгенерировать M-seq на CPU (из params_.lfsr_seed) и подготовить эталон.
  // Удобный shortcut для типового сценария.
  void PrepareReference();

  // Полный пайплайн корреляции: H2D(inp) → R2C → multiply_conj → C2R → extract.
  // inp: flat [S × N] float, row-major. Результат: peaks [S × K × n_kg].
  // PrepareReference() должен быть вызван ДО Process().
  FMCorrelatorResult Process(const std::vector<float>& inp);

  // Тестовый паттерн без H2D для входных сигналов: GPU-генерация
  // inp[s] = circshift(ref, s*shift_step). Ожидаемый пик сигнала s, сдвига k
  // находится в позиции (s*shift_step - k + N) % N.
  // PrepareReference() должен быть вызван ДО RunTestPattern().
  FMCorrelatorResult RunTestPattern(int shift_step = 2);

  // Автоматический батчинг для случая когда S не помещается в GPU-память.
  // total_signals может быть > params_.num_signals; BatchManager разобьёт.
  // Результат: объединённые peaks [total_signals × K × n_kg].
  FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp,
                                          int total_signals);

  const FMCorrelatorParams& GetParams() const { return params_; }

private:
  IBackend* backend_;           ///< Не владеем — lifecycle управляется снаружи
  FMCorrelatorParams params_;
  std::unique_ptr<FMCorrelatorProcessorROCm> processor_;  ///< Владеем — создан в конструкторе
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM

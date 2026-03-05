/**
 * @file fm_correlator.cpp
 * @brief FMCorrelator facade implementation
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

#include "fm_correlator.hpp"
#include <stdexcept>

namespace drv_gpu_lib {

FMCorrelator::FMCorrelator(IBackend* backend)
    : backend_(backend)
    , processor_(std::make_unique<FMCorrelatorProcessorROCm>(backend)) {
}

void FMCorrelator::SetParams(const FMCorrelatorParams& params) {
  params_ = params;
  processor_->SetParams(params);
}

std::vector<float> FMCorrelator::GenerateMSequence() const {
  return GenerateMSequence(params_.lfsr_seed);
}

/**
 * @brief Генерация M-последовательности на CPU через LFSR Галуа (Galois LFSR).
 *
 * Алгоритм: сдвигаем 32-битный регистр, старший бит извлекается как выход,
 * при bit=1 применяем XOR с полиномом (обратная связь). Бит → ±1 float.
 *
 * Почему Galois (а не Fibonacci): один XOR вместо нескольких → быстрее на CPU.
 * Почему CPU, а не GPU: генерируется только один раз, результат уходит в
 * PrepareReference() → H2D → apply_cyclic_shifts на GPU.
 *
 * @param seed Начальное состояние (должно быть != 0, иначе LFSR застывает)
 * @return float[fft_size] со значениями ±1.0f
 */
std::vector<float> FMCorrelator::GenerateMSequence(uint32_t seed) const {
  std::vector<float> seq(params_.fft_size);
  uint32_t lfsr = seed;
  for (size_t i = 0; i < params_.fft_size; ++i) {
    // Берём старший бит как выходной символ последовательности
    int bit = (lfsr >> 31) & 1;
    seq[i] = bit ? 1.0f : -1.0f;
    // При bit=1: сдвиг + XOR с полиномом (обратная связь по примитивным tap'ам)
    // При bit=0: просто сдвиг (без обратной связи)
    lfsr = bit ? ((lfsr << 1) ^ params_.lfsr_polynomial) : (lfsr << 1);
  }
  return seq;
}

void FMCorrelator::PrepareReference(const std::vector<float>& ref) {
  processor_->PrepareReference(ref);
}

void FMCorrelator::PrepareReference() {
  auto ref = GenerateMSequence();
  processor_->PrepareReference(ref);
}

FMCorrelatorResult FMCorrelator::Process(const std::vector<float>& inp) {
  return processor_->Process(inp);
}

FMCorrelatorResult FMCorrelator::RunTestPattern(int shift_step) {
  return processor_->RunTestPattern(shift_step);
}

FMCorrelatorResult FMCorrelator::ProcessWithBatching(
    const std::vector<float>& inp, int total_signals) {
  return processor_->ProcessWithBatching(inp, total_signals);
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM

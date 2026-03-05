#pragma once

/**
 * @file fm_correlator_types.hpp
 * @brief Types for FM Correlator: params + result
 *
 * FMCorrelatorParams  — FFT size, shifts, signals, LFSR config
 * FMCorrelatorResult  — peaks [S × K × n_kg] with accessor
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace drv_gpu_lib {

struct FMCorrelatorParams {
  // Размер FFT — ОБЯЗАТЕЛЬНО степень двойки: hipFFT требует для оптимального
  // плана. Определяет длину M-последовательности и временное разрешение.
  size_t fft_size = 32768;

  // K — число циклических сдвигов эталона, каждый сдвиг = одна гипотеза
  // задержки сигнала. Больше сдвигов → шире покрытие дальностей, но больше
  // GPU памяти (K×N float2 для ref_fft) и времени умножения.
  int num_shifts = 32;

  // S — число входных сигналов (лучей/антенн) за один вызов Process().
  // При S > допустимого — использовать ProcessWithBatching(), не увеличивать
  // num_signals вручную.
  int num_signals = 5;

  // n_kg — сколько первых точек брать из IFFT-результата корреляции.
  // Каждая точка j соответствует задержке j/fs секунд. Не нужно брать все N —
  // дальние задержки бессмысленны, а память экономим: peaks = S×K×n_kg < S×K×N.
  int num_output_points = 2000;

  // Примитивный полином LFSR степени 32 (x^32+x^22+x^2+x+1).
  // Определяет структуру M-последовательности: разные полиномы → разные псевдошумовые
  // свойства. Должен быть примитивным — иначе период < 2^32-1 и автокорреляция хуже.
  uint32_t lfsr_polynomial = 0x00400007;

  // Начальное состояние LFSR. При seed=0 LFSR зацикливается (нулевое состояние
  // поглощающее), поэтому seed ДОЛЖЕН быть ненулевым.
  uint32_t lfsr_seed = 0x12345678;
};

struct FMCorrelatorResult {
  // Выходной массив пиков корреляции, размер [S * K * n_kg], row-major:
  //   peaks[s*K*n_kg + k*n_kg + j] — значение для сигнала s, сдвига k, точки j.
  // Индексировать удобнее через at(signal, shift, point).
  std::vector<float> peaks;
  int num_signals = 0;
  int num_shifts = 0;
  int num_output_points = 0;

  // Безопасный доступ по трём индексам с inline-вычислением flat-offset.
  float at(int signal, int shift, int point) const {
    return peaks[static_cast<size_t>(
        (signal * num_shifts + shift) * num_output_points + point)];
  }
};

}  // namespace drv_gpu_lib

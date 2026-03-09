# FFT Processor — Tests

## Файлы тестов

| Файл | Namespace | Описание |
|------|-----------|----------|
| `test_fft_processor.hpp` | `test_fft_processor` | Базовые тесты FFTProcessor (OpenCL/clFFT) |
| `test_fft_vs_cpu.hpp` | `test_fft_vs_cpu` | Сравнение результатов GPU FFT с CPU (NumPy-эквивалент) |
| `test_fft_benchmark.hpp` | `test_fft_benchmark` | Бенчмарк FFTProcessor (OpenCL) через GpuBenchmarkBase |
| `test_fft_processor_rocm.hpp` | `test_fft_processor_rocm` | Базовые тесты FFTProcessorROCm (hipFFT) |
| `test_complex_to_mag_phase_rocm.hpp` | `test_complex_to_mag_phase_rocm` | Тест ComplexToMagPhaseROCm |
| `test_fft_benchmark_rocm.hpp` | `test_fft_benchmark_rocm` | Бенчмарк FFTProcessorROCm через GpuBenchmarkBase |
| `test_fft_matrix_rocm.hpp` | `test_fft_matrix_rocm` | Матричный бенчмарк: таблица beams × nFFT (ROCm) |
| `fft_processor_benchmark.hpp` | — | Benchmark-класс (OpenCL), наследник GpuBenchmarkBase |
| `fft_processor_benchmark_rocm.hpp` | — | Benchmark-класс (ROCm), наследник GpuBenchmarkBase |
| `all_test.hpp` | `fft_processor_all_test` | Точка входа: вызывается из `src/main.cpp` |

## Статус тестов

| Тест | Платформа | Статус |
|------|-----------|--------|
| `test_fft_processor` | OpenCL (clFFT) | ⏸ Отложен — clFFT не поддерживает gfx1201 (RDNA4) |
| `test_fft_vs_cpu` | OpenCL | ⏸ Отложен |
| `test_fft_benchmark` | OpenCL | ⏸ Отложен |
| `test_fft_processor_rocm` | ROCm (hipFFT) | ✅ Активен |
| `test_complex_to_mag_phase_rocm` | ROCm | ✅ Активен |
| `test_fft_benchmark_rocm` | ROCm | ⏸ Закомментирован |
| `test_fft_matrix_rocm` | ROCm | ✅ Активен |

## Запуск

Тесты вызываются через `fft_processor_all_test::run()` из `src/main.cpp`.
Включить/выключить тест: отредактировать `all_test.hpp`.

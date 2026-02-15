#include <iostream>

#include "DrvGPU/tests/single_gpu.hpp"
//#include "DrvGPU/tests/example_external_context_usage.hpp"
//#include "modules/example/tests/test_vector_ops.hpp"
//#include "modules/search_maxim/tests/test_antenna_module.hpp"
//#include "modules/fft_maxima/tests/test_fft_maxima.hpp"
#include "modules/fft_maxima/tests/test_spectrum_maxima.hpp"
#include "modules/fft_maxima/tests/test_large_batch.hpp"
#include "modules/fft_maxima/tests/test_gpu_generator_integration.hpp"
#include "modules/signal_generators/tests/test_signal_generators.hpp"
#include "modules/fft_processor/tests/test_fft_processor.hpp"
#include "modules/fft_processor/tests/test_fft_vs_cpu.hpp"
#include "modules/fft_maxima/tests/test_find_all_maxima.hpp"
#include "modules/fft_maxima/tests/test_benchmark_all_maxima.hpp"
#include "modules/fft_maxima/tests/test_batch_all_maxima.hpp"
//#include "DrvGPU/tests/test_services.hpp"
//#include "DrvGPU/tests/test_gpu_profiler.hpp"

//int main(int argc, char* argv[]) {
int main() {
   std::cout << "===============================================================\n"
            << "Набор библиотек для работы с GPU\n"
            << "===============================================================\n\n";
  std::cout << "Программа успешно запущена!" << std::endl;

//  example_drv_gpu_singl::run();

//  example_drv_gpu_multi::run();
//  external_context_example::run();

//  test_example_mat::run();
//  test_find_3_max::run();
//  test_fft_max::run();
//  test_spectrum_maxima::run();
//  test_large_batch::run();  // НОВЫЙ API с batch processing
  
// !!! новые нужно разбераться !!!!!!
// test_gpu_generator_integration::run();  // Новый API (GPU→GPU) ПЕРЕГИБ 
// !!! новые нужно разбераться !!!!!!
//  test_signal_generators::run();  // Signal Generators: CW, LFM, Noise
// !!! новые нужно разбераться !!!!!!
//  test_fft_processor::run();  // FFTProcessor: FFT с разными режимами вывода
// !!! новые нужно разбераться !!!!!!
//  test_fft_vs_cpu::run();     // FFTProcessor vs CPU reference (pocketfft)
// !!! новые нужно разбераться !!!!!!
// test_find_all_maxima::run();  // FindAllMaxima: поиск всех максимумов
// test_batch_all_maxima::run();  // BATCH: тесты для batch-обработки
  test_benchmark_all_maxima::run();  // BENCHMARK: 10 лучей × 500k точек


  // Services multithreaded tests (без теста профилирования — он отдельно)
//  test_services::run();

  // Отдельный тест GPUProfiler: многопоточный Record, агрегация, PrintSummary
//  test_gpu_profiler::run();

   std::cout << "\nВсе тесты завершены!" << std::endl;
  return 0;
}


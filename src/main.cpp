#include <iostream>

#include "DrvGPU/tests/single_gpu.hpp"
#include "DrvGPU/tests/multi_gpu.hpp"
#include "DrvGPU/tests/example_external_context_usage.hpp"
#include "modules/example/tests/test_vector_ops.hpp"
//#include "modules/search_maxim/tests/test_antenna_module.hpp"
//#include "modules/fft_maxima/tests/test_fft_maxima.hpp"
#include "modules/fft_maxima/tests/test_spectrum_maxima.hpp"
#include "DrvGPU/tests/test_services.hpp"

//int main(int argc, char* argv[]) {
int main() {
   std::cout << "═══════════════════════════════════════════════════════════\n"
            << "Набор библиотек для работы с GPU\n"
            << "═══════════════════════════════════════════════════════════\n\n";
  std::cout << "✅ Программа успешно запущена!" << std::endl;

//  example_drv_gpu_singl::run();
  
//  example_drv_gpu_multi::run();
//  external_context_example::run();

//  test_example_mat::run();  
//  test_find_3_max::run();
//  test_fft_max::run();
  test_spectrum_maxima::run();

  // Services multithreaded tests
  test_services::run();

   std::cout << "\n✅ Все тесты завершены!" << std::endl;
  return 0;
}


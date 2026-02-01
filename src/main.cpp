#include <iostream>
#include <string>

#include "../tests/single_gpu.hpp"
#include "../tests/multi_gpu.hpp"
#include "../tests/example_external_context_usage.hpp"

int main(int argc, char* argv[]) {
   std::cout << "═══════════════════════════════════════════════════════════\n"
            << "Набор библиотек для работы с GPU\n"
            << "═══════════════════════════════════════════════════════════\n\n";
  std::cout << "✅ Программа успешно запущена!" << std::endl;

  example_drv_gpu_singl::run();
  example_drv_gpu_multi::run();
  external_context_example::run();

  
  return 0;
}


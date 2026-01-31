#include <iostream>
#include <string>

#include "single_gpu.hpp"


int main(int argc, char* argv[]) {
   std::cout << "═══════════════════════════════════════════════════════════\n"
            << "Набор библиотек для работы с GPU\n"
            << "═══════════════════════════════════════════════════════════\n\n";
  std::cout << "✅ Программа успешно запущена!" << std::endl;
example_drv_gpu_singl::run();
//example_drv_gpu_multi::run();
  return 0;
}


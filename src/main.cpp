#include <iostream>

#include "../tests/single_gpu.hpp"
#include "../tests/multi_gpu.hpp"
#include "../tests/example_external_context_usage.hpp"
#include "../tests/test_vector_ops.hpp"

//int main(int argc, char* argv[]) {
int main() {
   std::cout << "═══════════════════════════════════════════════════════════\n"
            << "Набор библиотек для работы с GPU\n"
            << "═══════════════════════════════════════════════════════════\n\n";
  std::cout << "✅ Программа успешно запущена!" << std::endl;

//  example_drv_gpu_singl::run();

  
//  example_drv_gpu_multi::run();
//  external_context_example::run();

  test_example_mat::run();  
  return 0;
}


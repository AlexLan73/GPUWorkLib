#include <iostream>

// Вызов тестов через all_test.hpp каждого модуля (см. CLAUDE.md)
#include "DrvGPU/tests/all_test.hpp"
#include "modules/fft_maxima/tests/all_test.hpp"
#include "modules/fft_processor/tests/all_test.hpp"
#include "modules/signal_generators/tests/all_test.hpp"
#include "modules/lch_farrow/tests/all_test.hpp"
#include "modules/filters/tests/all_test.hpp"
#include "modules/heterodyne/tests/all_test.hpp"
#include "modules/statistics/tests/all_test.hpp"
#include "modules/vector_algebra/tests/all_test.hpp"

int main() {
    std::cout << "===============================================================\n"
              << "Набор библиотек для работы с GPU\n"
              << "===============================================================\n\n";
    std::cout << "Программа успешно запущена!" << std::endl;

    // DrvGPU: примеры и тесты
//    drvgpu_all_test::run();

    // fft_maxima: SpectrumMaximaFinder
//    fft_maxima_all_test::run();

    // fft_processor: FFT с режимами Complex/MagPhase
    fft_processor_all_test::run();

    // signal_generators: CW, LFM, Noise, FormSignal, FormScript (SaveKernel/LoadKernel)
//    signal_generators_all_test::run();

    // lch_farrow: standalone Lagrange fractional delay
//    lch_farrow_all_test::run();

    // filters: FIR + IIR GPU filters
//    filters_all_test::run();

    // heterodyne: LFM dechirp processing
//    heterodyne_all_test::run();

    // statistics: mean, median, variance, std (ROCm only)
//    statistics_all_test::run();

    // vector_algebra: Cholesky matrix inversion (ROCm only)
//    vector_algebra_all_test::run();

    std::cout << "\nВсе тесты завершены!" << std::endl;
    return 0;
}

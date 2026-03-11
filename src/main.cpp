#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cctype>

// Вызов тестов через all_test.hpp каждого модуля (см. CLAUDE.md)
#include "DrvGPU/tests/all_test.hpp"
#include "modules/fft_func/tests/all_test.hpp"
#include "modules/signal_generators/tests/all_test.hpp"
#include "modules/lch_farrow/tests/all_test.hpp"
#include "modules/filters/tests/all_test.hpp"
#include "modules/heterodyne/tests/all_test.hpp"
#include "modules/statistics/tests/all_test.hpp"
#include "modules/vector_algebra/tests/all_test.hpp"
#include "modules/fm_correlator/tests/all_test.hpp"
#include "modules/strategies/tests/all_test.hpp"

namespace {

// Порядок модулей для режима "all" (MemoryBank/specs/create_agent_test.md)
const char* kDefaultOrder[] = {
    "drvgpu", "fft_func", "statistics", "vector_algebra",
    "filters", "signal_generators", "lch_farrow", "heterodyne",
    "fm_correlator", "strategies"
};
const size_t kDefaultOrderSize = sizeof(kDefaultOrder) / sizeof(kDefaultOrder[0]);

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::vector<std::string> read_modules_from_file(const std::string& path) {
    std::vector<std::string> modules;
    std::ifstream f(path);
    if (!f) return modules;

    std::string line;
    while (std::getline(f, line)) {
        // Strip comment (include end-of-line: "module  # skip")
        size_t hash = line.find('#');
        if (hash != std::string::npos)
            line.resize(hash);

        // Trim
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t");
        std::string trimmed = line.substr(start, end - start + 1);

        // Skip empty
        if (trimmed.empty()) continue;

        modules.push_back(to_lower(trimmed));
    }
    return modules;
}

bool run_module(const std::string& name) {
    std::string n = to_lower(name);

    if (n == "drvgpu")           { drvgpu_all_test::run(); return true; }
    if (n == "fft_func")         { fft_func_all_test::run(); return true; }
    if (n == "statistics")       { statistics_all_test::run(); return true; }
    if (n == "vector_algebra")   { vector_algebra_all_test::run(); return true; }
    if (n == "filters")          { filters_all_test::run(); return true; }
    if (n == "signal_generators"){ signal_generators_all_test::run(); return true; }
    if (n == "lch_farrow")       { lch_farrow_all_test::run(); return true; }
    if (n == "heterodyne")       { heterodyne_all_test::run(); return true; }
    if (n == "fm_correlator")    { fm_correlator_all_test::run(); return true; }
    if (n == "strategies")       { strategies_all_test::run(); return true; }

    return false;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [<file> | all | <module> | --file <path>]\n"
              << "  (no args)    - run modules from config/tests_order.txt\n"
              << "                Use # to comment out modules to skip\n"
              << "  all          - same as no args (config file or default order)\n"
              << "  <module>     - run single module\n"
              << "  --file path  - run modules from specified file\n"
              << "Modules: drvgpu, fft_func, statistics, vector_algebra,\n"
              << "         filters, signal_generators, lch_farrow, heterodyne,\n"
              << "         fm_correlator, strategies\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::cout << "===============================================================\n"
              << "Набор библиотек для работы с GPU\n"
              << "===============================================================\n\n";

    std::vector<std::string> to_run;

    if (argc == 1) {
        // No args: read config/tests_order.txt (comment out lines to skip)
        auto modules = read_modules_from_file("config/tests_order.txt");
        if (modules.empty())
            modules = read_modules_from_file("../config/tests_order.txt");
        if (modules.empty())
            to_run.assign(kDefaultOrder, kDefaultOrder + kDefaultOrderSize);
        else
            to_run = std::move(modules);
    } else if (argc >= 2) {
        std::string arg1(argv[1]);
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg1 == "all") {
            // Try config/tests_order.txt (from project root) or ../config/
            auto modules = read_modules_from_file("config/tests_order.txt");
            if (modules.empty())
                modules = read_modules_from_file("../config/tests_order.txt");
            if (modules.empty())
                to_run.assign(kDefaultOrder, kDefaultOrder + kDefaultOrderSize);
            else
                to_run = std::move(modules);
        } else if (arg1 == "--file" && argc >= 3) {
            to_run = read_modules_from_file(argv[2]);
            if (to_run.empty()) {
                std::cerr << "No modules in file or file not found: " << argv[2] << "\n";
                return 1;
            }
        } else {
            to_run.push_back(to_lower(arg1));
        }
    }

    if (to_run.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Modules to run: ";
    for (size_t i = 0; i < to_run.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << to_run[i];
    }
    std::cout << " (" << to_run.size() << ")\n\n";

    for (const auto& m : to_run) {
        std::cout << "\n>>> " << m << " <<<\n";
        if (!run_module(m)) {
            std::cerr << "Unknown module: " << m << "\n";
            return 1;
        }
    }

    std::cout << "\nВсе тесты завершены!" << std::endl;
    return 0;
}

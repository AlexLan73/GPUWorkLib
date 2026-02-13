#pragma once

/**
 * @file system_sampling.hpp
 * @brief Системные параметры дискретизации
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include <cstddef>

namespace signal_gen {

/**
 * @brief Параметры дискретизации — общие для всех генераторов
 */
struct SystemSampling {
    double fs = 1000.0;    ///< Частота дискретизации (Hz)
    size_t length = 1024;  ///< Количество отсчётов на луч
};

} // namespace signal_gen

#pragma once

/**
 * @file prof_utils.hpp
 * @brief CollectOrRelease — общая утилита профилирования OpenCL событий
 *
 * Используется во всех генераторах модуля signal_generators.
 * Предотвращает дублирование кода в каждом .cpp файле.
 *
 * Правило: вызывать ПОСЛЕ того как event использован как wait-dependency.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-09
 */

#include <CL/cl.h>
#include <utility>
#include <vector>

namespace signal_gen {

/**
 * @brief Сохранить cl_event для профилирования или освободить (production path)
 *
 * @tparam ProfEvents  Тип вектора событий (vector<pair<const char*, cl_event>>)
 * @param ev           OpenCL event (может быть nullptr — тогда ничего не делает)
 * @param name         Имя стадии (строковый литерал)
 * @param prof_events  nullptr → production (clReleaseEvent); иначе → сохранить в вектор
 */
template <typename ProfEvents>
inline void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events) {
  if (!ev) return;
  if (prof_events)
    prof_events->push_back({name, ev});
  else
    clReleaseEvent(ev);
}

}  // namespace signal_gen

#pragma once
/**
 * @file opencl_profiling.hpp
 * @brief Хелпер: заполнение OpenCLProfilingData из cl_event (5 параметров cl_profiling_info)
 */

#include <CL/cl.h>
#include "../../services/gpu_profiler.hpp"

namespace drv_gpu_lib {

/**
 * @brief Заполнить OpenCLProfilingData из cl_event (все 5 параметров)
 * @param event cl_event с включённым profiling (CL_QUEUE_PROFILING_ENABLE)
 * @param out Заполняемая структура
 * @return true если все 5 значений получены, false при ошибке
 */
bool FillOpenCLProfilingData(cl_event event, OpenCLProfilingData& out);

} // namespace drv_gpu_lib

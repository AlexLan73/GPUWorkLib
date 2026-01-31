#pragma once
#include <cstdint>
#include <stdint.h>
// ═════════════════════════════════════════════════════════════════════
// DELAY PARAMETER STRUCTURE
// ═════════════════════════════════════════════════════════════════════

typedef struct
{
  size_t beam_index; // Индекс луча [0...num_beams)
  float delay_degrees; // Задержка от УГЛА (градусы)
  float delay_time_ns; // Задержка по ВРЕМЕНИ (наносекунды)
} CombinedDelayParam;

#pragma once

/**
 * @file null_checkpoint_save.hpp
 * @brief NullCheckpointSave - Null Object pattern (zero overhead in production)
 *
 * All methods are empty inline no-ops. Compiler can eliminate vtable calls
 * when devirtualization is possible.
 *
 * @date 2026-03-07
 */

#include "interfaces/i_checkpoint_save.hpp"

namespace strategies {

class NullCheckpointSave final : public ICheckpointSave {
public:
  void save_c1_signal(const void*, uint32_t, uint32_t, float, int) override {}
  void save_c1_weights(const void*, uint32_t, int) override {}
  void save_c2_data(const void*, uint32_t, uint32_t, float, int) override {}
  void save_c2_stats(const statistics::StatisticsResult*, const statistics::StatisticsResult*,
                     uint32_t, int) override {}
  void save_c3_spectrum(const void*, uint32_t, uint32_t, int) override {}
  void save_c3_minmax(const MinMaxResult*, uint32_t, int) override {}
  void save_c4_one_max(const OneMaxParabolaLite*, uint32_t, int) override {}
};

}  // namespace strategies

#pragma once

/**
 * @file branch_selector.hpp
 * @brief BranchSelector — stateful hysteresis selector для SNR-estimator
 *
 * SNR_05: отдельный класс от StatisticsProcessor (SOLID).
 * Facade остаётся stateless; hysteresis state живёт здесь.
 *
 * Usage:
 *   statistics::BranchSelector selector;
 *   while (true) {
 *     auto r = processor.ComputeSnrDb(data, ...);
 *     auto branch = selector.Select(r.snr_db_global, cfg.thresholds);
 *     dispatch_to_branch(branch);
 *   }
 *
 * NOT thread-safe — один экземпляр на поток/pipeline.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#include "statistics_types.hpp"

#include <cmath>  // std::isfinite

namespace statistics {

/**
 * @brief Stateful branch selector with hysteresis.
 *
 * Применяется после `StatisticsProcessor::ComputeSnrDb()`. Хранит текущую
 * ветку между вызовами. Переход происходит только когда SNR пересекает
 * порог + hysteresis (чтобы подавить дребезг на границе).
 *
 * Переходы:
 *   Low  → Mid:  snr > low_to_mid + h
 *   Mid  → Low:  snr < low_to_mid - h
 *   Mid  → High: snr > mid_to_high + h
 *   High → Mid:  snr < mid_to_high - h
 *
 * Защита от NaN/Inf: невалидное измерение не меняет state (оставляем текущую).
 */
class BranchSelector {
public:
  BranchSelector() = default;

  /**
   * @brief Select branch for given SNR with hysteresis. Updates state.
   *
   * @param snr_db  Измеренный SNR_db (от ComputeSnrDb)
   * @param thr     Пороги (из SnrEstimationConfig::thresholds)
   * @return Новая (или прежняя, если не пересекли) ветка
   */
  BranchType Select(float snr_db, const BranchThresholds& thr) {
    // Защита от NaN/Inf: невалидное измерение → оставляем текущую ветку
    // (ломать переключение одним плохим фреймом нельзя).
    if (!std::isfinite(snr_db)) {
      return current_;
    }
    const float h = thr.hysteresis_db;
    switch (current_) {
      case BranchType::Low:
        if (snr_db > thr.low_to_mid_db + h) {
          current_ = BranchType::Mid;
        }
        break;
      case BranchType::Mid:
        if (snr_db > thr.mid_to_high_db + h) {
          current_ = BranchType::High;
        } else if (snr_db < thr.low_to_mid_db - h) {
          current_ = BranchType::Low;
        }
        break;
      case BranchType::High:
        if (snr_db < thr.mid_to_high_db - h) {
          current_ = BranchType::Mid;
        }
        break;
    }
    return current_;
  }

  /// Get current branch without updating state
  BranchType Current() const { return current_; }

  /// Reset state (e.g. at start of new measurement session)
  void Reset(BranchType to = BranchType::Low) { current_ = to; }

private:
  BranchType current_ = BranchType::Low;
};

}  // namespace statistics

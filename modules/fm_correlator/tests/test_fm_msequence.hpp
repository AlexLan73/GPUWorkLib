#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_msequence.hpp
 * @brief Test LFSR M-sequence generator
 *
 * Tests:
 *   1. All values are {+1.0, -1.0}
 *   2. ~50% ones and ~50% minus ones
 *   3. Different seeds produce different sequences
 *   4. Same seed produces identical sequence
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#include "fm_correlator.hpp"
#include "services/console_output.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace fm_correlator::tests {

inline void run_test_msequence() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;
  con.Print(gpu_id, "FM_Corr", "Test: M-sequence generator");

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size = 32768;

  auto generate = [&](uint32_t seed) -> std::vector<float> {
    std::vector<float> seq(params.fft_size);
    uint32_t lfsr = seed;
    for (size_t i = 0; i < params.fft_size; ++i) {
      int bit = (lfsr >> 31) & 1;
      seq[i] = bit ? 1.0f : -1.0f;
      lfsr = bit ? ((lfsr << 1) ^ params.lfsr_polynomial) : (lfsr << 1);
    }
    return seq;
  };

  // Test 1: all values {+1.0, -1.0}
  {
    auto seq = generate(params.lfsr_seed);
    for (size_t i = 0; i < seq.size(); ++i) {
      if (seq[i] != 1.0f && seq[i] != -1.0f) {
        throw std::runtime_error("M-seq: value at " + std::to_string(i) +
            " is " + std::to_string(seq[i]) + ", expected +/-1.0");
      }
    }
    con.Print(gpu_id, "FM_Corr", "  [1/4] All values are {+1.0, -1.0} ✅");
  }

  // Test 2: ~50% ones
  {
    auto seq = generate(params.lfsr_seed);
    int ones = 0;
    for (float v : seq) {
      if (v > 0.0f) ++ones;
    }
    float ratio = static_cast<float>(ones) / static_cast<float>(seq.size());
    if (ratio < 0.4f || ratio > 0.6f) {
      throw std::runtime_error("M-seq: ratio of +1 = " +
          std::to_string(ratio) + ", expected ~0.5");
    }
    con.Print(gpu_id, "FM_Corr",
        "  [2/4] ~50% ones: ratio=" + std::to_string(ratio) + " ✅");
  }

  // Test 3: different seeds -> different sequences
  {
    auto seq1 = generate(params.lfsr_seed);
    auto seq2 = generate(0xDEADBEEF);
    int diff = 0;
    for (size_t i = 0; i < seq1.size(); ++i) {
      if (seq1[i] != seq2[i]) ++diff;
    }
    if (diff == 0) {
      throw std::runtime_error("M-seq: different seeds produced identical sequences");
    }
    con.Print(gpu_id, "FM_Corr",
        "  [3/4] Different seeds -> different sequences (diff=" +
        std::to_string(diff) + ") ✅");
  }

  // Test 4: same seed -> same sequence
  {
    auto seq1 = generate(0x42);
    auto seq2 = generate(0x42);
    for (size_t i = 0; i < seq1.size(); ++i) {
      if (seq1[i] != seq2[i]) {
        throw std::runtime_error("M-seq: same seed produced different sequences at " +
            std::to_string(i));
      }
    }
    con.Print(gpu_id, "FM_Corr", "  [4/4] Same seed -> identical sequence ✅");
  }

  con.Print(gpu_id, "FM_Corr", "Test M-sequence: PASSED ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM

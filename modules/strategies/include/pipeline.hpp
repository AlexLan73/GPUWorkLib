#pragma once

/**
 * @file pipeline.hpp
 * @brief Pipeline — ordered execution of IPipelineStep with parallel groups
 *
 * Ref03-C: Holds steps in execution order. Supports:
 * - Sequential steps (one at a time)
 * - Parallel groups (steps on different streams, sync after group)
 * - FindStep(name) for test access to individual steps
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "i_pipeline_step.hpp"
#include "pipeline_context.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <memory>
#include <string>
#include <cstring>

namespace strategies {

/// Group of steps executed in parallel on different streams
struct ParallelGroup {
  std::vector<IPipelineStep*> steps;     ///< non-owning
  std::vector<hipStream_t>    streams;   ///< one per step
};

class Pipeline {
public:
  /// Entry in execution order
  struct Entry {
    enum Type { SEQUENTIAL, PARALLEL };
    Type type = SEQUENTIAL;
    IPipelineStep* step = nullptr;         ///< for SEQUENTIAL
    size_t parallel_group_index = 0;       ///< for PARALLEL
  };

  /// Execute all steps in order
  void Execute(PipelineContext& ctx) {
    for (auto& entry : entries_) {
      if (entry.type == Entry::SEQUENTIAL) {
        if (entry.step->IsEnabled(*ctx.cfg)) {
          entry.step->Execute(ctx);
        }
      } else {
        auto& group = parallel_groups_[entry.parallel_group_index];
        for (size_t i = 0; i < group.steps.size(); ++i) {
          if (group.steps[i]->IsEnabled(*ctx.cfg)) {
            group.steps[i]->Execute(ctx);
          }
        }
        // Sync all parallel streams
        for (auto stream : group.streams) {
          hipStreamSynchronize(stream);
        }
      }
    }
  }

  /// Find step by name (for AntennaProcessorTest direct step execution)
  IPipelineStep* FindStep(const char* name) const {
    for (auto& s : all_steps_) {
      if (std::strcmp(s->Name(), name) == 0) return s.get();
    }
    return nullptr;
  }

private:
  friend class PipelineBuilder;

  std::vector<std::unique_ptr<IPipelineStep>> all_steps_;   ///< ownership
  std::vector<Entry> entries_;                               ///< execution order
  std::vector<ParallelGroup> parallel_groups_;
};

}  // namespace strategies

#endif  // ENABLE_ROCM

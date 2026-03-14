#pragma once

/**
 * @file pipeline_builder.hpp
 * @brief PipelineBuilder — fluent API for constructing Pipeline
 *
 * Ref03-C: Build pipelines declaratively:
 *   auto pipe = PipelineBuilder()
 *     .add(make_unique<GemmStep>())
 *     .add(make_unique<WindowFftStep>())
 *     .add_parallel({OneMax, AllMaxima, MinMax}, {s1, s2, s3})
 *     .build();
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "pipeline.hpp"

#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

namespace strategies {

class PipelineBuilder {
public:
  /// Add a sequential step
  PipelineBuilder& add(std::unique_ptr<IPipelineStep> step) {
    Pipeline::Entry entry;
    entry.type = Pipeline::Entry::SEQUENTIAL;
    entry.step = step.get();
    entries_.push_back(entry);
    steps_.push_back(std::move(step));
    return *this;
  }

  /// Add a step only if condition is true
  PipelineBuilder& add_if(bool condition, std::unique_ptr<IPipelineStep> step) {
    if (condition) return add(std::move(step));
    return *this;
  }

  /// Add a parallel group (steps run on different streams concurrently)
  PipelineBuilder& add_parallel(
      std::vector<std::unique_ptr<IPipelineStep>> group_steps,
      std::vector<hipStream_t> streams) {
    ParallelGroup pg;
    Pipeline::Entry entry;
    entry.type = Pipeline::Entry::PARALLEL;
    entry.parallel_group_index = parallel_groups_.size();

    for (size_t i = 0; i < group_steps.size(); ++i) {
      pg.steps.push_back(group_steps[i].get());
      if (i < streams.size()) pg.streams.push_back(streams[i]);
    }

    entries_.push_back(entry);
    for (auto& s : group_steps) steps_.push_back(std::move(s));
    parallel_groups_.push_back(std::move(pg));
    return *this;
  }

  /// Build the pipeline (moves ownership)
  std::unique_ptr<Pipeline> build() {
    auto p = std::make_unique<Pipeline>();
    p->all_steps_ = std::move(steps_);
    p->entries_ = std::move(entries_);
    p->parallel_groups_ = std::move(parallel_groups_);
    return p;
  }

private:
  std::vector<std::unique_ptr<IPipelineStep>> steps_;
  std::vector<Pipeline::Entry> entries_;
  std::vector<ParallelGroup> parallel_groups_;
};

}  // namespace strategies

#endif  // ENABLE_ROCM

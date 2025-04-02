#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khovansky_d_double_radix_batcher_seq {

class RadixSeq : public ppc::core::Task {
 public:
  explicit RadixSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
};
}  // namespace khovansky_d_double_radix_batcher_seq
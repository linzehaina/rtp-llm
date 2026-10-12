#pragma once

#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include <utility>
#include <vector>

namespace rtp_llm::detail {

// CUDA sampling reserves RNG for every dense row, including skipped rows.
// The two contiguous output views own one fresh allocation; neither is reused.
std::pair<torch::Tensor, torch::Tensor> makeSamplingSeedOffsetTensors(
    const std::vector<at::Generator>& generators, int64_t batch_size, int increment, TensorHolder* buffer_holder);

}  // namespace rtp_llm::detail

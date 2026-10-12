#pragma once

#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include <vector>

namespace rtp_llm {

// Preserve device-authoritative lengths for existing requests. Stage missing
// rows together, so a new cohort does not issue one blocking scalar H2D per row.
inline torch::Tensor stagePreviousSequenceLengths(const std::vector<torch::Tensor>& device_lengths,
                                                   const std::vector<int32_t>& host_lengths,
                                                   const torch::TensorOptions& device_options,
                                                   TensorHolder& host_holder) {
    TORCH_CHECK(device_lengths.size() == host_lengths.size(), "sequence-length row counts differ");
    const auto batch_size = static_cast<int64_t>(device_lengths.size());
    if (batch_size == 0) {
        return torch::empty({0}, device_options);
    }

    std::vector<torch::Tensor> rows;
    rows.reserve(device_lengths.size());
    size_t missing = 0;
    for (const auto& length : device_lengths) {
        if (length.defined() && length.is_cuda()) {
            rows.push_back(length.reshape({1}));
        } else {
            rows.emplace_back();
            ++missing;
        }
    }
    if (missing == 0) {
        return torch::cat(rows, 0);
    }

    auto host = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    auto* values = host.data_ptr<int32_t>();
    for (int64_t row = 0; row < batch_size; ++row) {
        values[row] = rows[row].defined() ? 0 : host_lengths[row];
    }
    host_holder.hold_host(host);
    auto staged = host.to(device_options, /*non_blocking=*/true);
    if (missing == device_lengths.size()) {
        return staged;
    }
    for (int64_t row = 0; row < batch_size; ++row) {
        if (!rows[row].defined()) {
            rows[row] = staged.narrow(0, row, 1);
        }
    }
    return torch::cat(rows, 0);
}

}  // namespace rtp_llm

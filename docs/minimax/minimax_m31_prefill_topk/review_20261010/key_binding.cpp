#include <torch/extension.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/minimax_prefill_topk.h"
namespace torch_ext { torch::Tensor probe_score_keys(const torch::Tensor& input); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("minimax_prefill_topk", &torch_ext::minimax_prefill_topk);
    m.def("probe_score_keys", &torch_ext::probe_score_keys);
}

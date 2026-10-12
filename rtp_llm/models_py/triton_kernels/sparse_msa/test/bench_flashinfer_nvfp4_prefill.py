"""Benchmark the RTP-to-FlashInfer prefill adapter on request-local CP4 pages."""

import argparse
import json
import statistics

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.flashinfer_nvfp4_prefill import (
    _PLANAR_PAGES,
    _convert_page_head,
    flashinfer_sparse_prefill_from_topk_fp4,
)
from flashinfer.msa_ops import _nvfp4_prefill_sm100 as fi


def measure(graph, repeats=10):
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / repeats)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main(batch, history):
    device = torch.device("cuda:0")
    prefix = (history * 9 // 10 // 128) * 128
    half_q = (history - prefix) // 8
    assert half_q > 0 and (history - prefix) % 8 == 0
    segments = batch * 2
    rows = segments * half_q
    pages_per_request = (history + 127) // 128
    pages = batch * pages_per_request
    shape = (pages, 4, 128, 64)
    values = torch.randint(0, 256, (2,) + shape, device=device, dtype=torch.uint8)
    scales = torch.randint(0x20, 0x48, (2, pages, 4 * 128 * 8),
                           device=device, dtype=torch.uint8)
    q = torch.randn((rows, 64, 128), device=device, dtype=torch.bfloat16)
    seq_host = [prefix + half_q if seg % 2 == 0 else history
                for seg in range(segments)]
    offset_host = [prefix if seg % 2 == 0 else history - half_q
                   for seg in range(segments)]
    seqused_k = torch.tensor(seq_host, device=device, dtype=torch.int32)
    full_lengths = torch.full((batch,), history, device=device, dtype=torch.int32)
    positions = torch.cat([
        torch.arange(start, start + half_q, device=device, dtype=torch.int32)
        for start in offset_host
    ])
    cu_q = torch.arange(segments + 1, device=device, dtype=torch.int32) * half_q
    req_to_token = (torch.arange(pages_per_request * 128, device=device,
                                 dtype=torch.int32).view(1, -1)
                    + torch.arange(batch, device=device, dtype=torch.int32)
                    .view(-1, 1) * pages_per_request * 128)
    req_to_token = req_to_token.repeat_interleave(2, dim=0)
    topk = torch.empty((4, rows, 16), device=device, dtype=torch.int32)
    for seg, length in enumerate(seq_host):
        chosen = torch.linspace(0, (length - 1) // 128, 16,
                                device=device).to(torch.int32)
        topk[:, seg * half_q:(seg + 1) * half_q] = chosen.flip(0)

    fi.warm(device)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    def run():
        return flashinfer_sparse_prefill_from_topk_fp4(
            q, values[0], values[1], scales[0], scales[1], topk,
            req_to_token, cu_q, seqused_k, positions, full_lengths,
            128**-0.5, segments_per_request=2)

    output = run()
    pool, layout = _PLANAR_PAGES.acquire(pages, 4, device)

    def convert():
        _convert_page_head[(pages, 4)](
            values[0], values[1], scales[0], scales[1], pool,
            values[0].stride(0), values[1].stride(0),
            scales[0].stride(0), scales[1].stride(0),
            PAGE_BYTES=layout["page_bytes"],
            K_SCALE_OFFSET=layout["k_scale_byte_offset"],
            V_DATA_OFFSET=layout["v_data_byte_offset"],
            V_SCALE_OFFSET=layout["v_scale_byte_offset"], num_warps=8)

    convert_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(convert_graph):
        convert()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    print(json.dumps({
        "batch": batch, "history": history, "prefix": prefix,
        "cp": 4, "rank_q": rows, "working_pages": pages,
        "adapter": measure(graph), "convert": measure(convert_graph),
        "extra_allocated_mib": (torch.cuda.memory_allocated(device) - baseline) / 2**20,
        "extra_peak_mib": (torch.cuda.max_memory_allocated(device) - baseline) / 2**20,
        "planar_pool_mib": pages * fi.page_layout(4)["page_bytes"] / 2**20,
        "finite": bool(torch.isfinite(output).all().item()),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--history", type=int, default=74000)
    args = parser.parse_args()
    main(args.batch, args.history)

"""Measure the NPU compute and communication costs used by Test 5."""

import math
import os
import sys
from pathlib import Path

import torch
import torch_npu  # Registers the NPU backend.
from torch_npu.contrib import transfer_to_npu  # Adapts legacy Megatron CUDA calls.
from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
from torch.distributed.elastic.multiprocessing.errors import record

MICRO_BATCH = 2
HIDDEN = 4096


@record
def main():
    action, path, length = sys.argv[1:]
    profile = Path(path)
    sequence = int(length)
    activation = MICRO_BATCH * sequence * HIDDEN
    reductions = (2, MICRO_BATCH * sequence, 65_536, activation, 2**24, 2**26, HIDDEN * 50_000)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    heads = [16, 32]
    hidden_dims = [2048, HIDDEN, 8192, 16384]
    configs = OfflineProfileConfigs(
        device=f"npu:{local_rank}",
        dtype=torch.float16,
        min_batch_size=MICRO_BATCH,
        max_batch_size=MICRO_BATCH,
        min_seq_len=sequence,
        max_seq_len=sequence,
        num_warmup_steps=3,
        num_profile_steps=7,
        force_overwrite=(action in ("all_reduce", "p2p")),
        use_torch_empty_to_init_weight=False,
        use_many_to_many_p2p_offline_data=True,
        many_to_many_p2p_rank_mapping={0: 1, 1: 0},
        many_to_many_sender_ranks=[0],
        many_to_many_recver_ranks=[1],
        comm_data_sizes=list(reductions if action == "all_reduce" else [activation]),
        gemm_left_op_hidden_dims=hidden_dims,
        gemm_right_op_hidden_dims=[2048, HIDDEN, 6144, 8192, 12288, 16384, 25000, 50000],
        bmm_op_hidden_dims=[2048, HIDDEN],
        baddbmm_op_hidden_dims=[2048, HIDDEN],
        num_attn_heads_per_rank=heads,
        layernorm_op_hidden_dims=hidden_dims,
        dropout_op_hidden_dims=hidden_dims,
        dropout_op_num_attn_heads=heads,
        masked_fill_op_num_attn_heads=heads,
        softmax_op_num_attn_heads=heads,
    )
    profiler = CriusOfflineDataManager(configs, str(profile.parent), profile.name)
    if action in ("all_reduce", "p2p"):
        try:
            profiler.profile(action)
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        return

    records = profiler.recorder.data
    required = [f"all_reduce__n1__gs2__d{n}" for n in reductions]
    for key in required:
        latency = records.get(key, {}).get("latency_ms")
        if not isinstance(latency, (float, int)) or not 0 < latency < math.inf:
            raise SystemExit(f"Missing or invalid single-node two-NPU profile for {key}. Check the communication profiling logs.")
    p2p_key = f"p2p__n1__gs2__d{activation}"
    latency = records.get(p2p_key, {}).get("latency_ms")
    if not isinstance(latency, list) or len(latency) != 3 or any(
        not 0 < value < math.inf for value in latency
    ):
        raise SystemExit(f"Missing or invalid single-node two-NPU profile for {p2p_key}. Check the communication profiling logs.")

    if action not in ("compute", "check"):
        raise ValueError(f"Unknown profiling action {action}.")
    for opcode in ("gemm", "bmm", "baddbmm", "layernorm", "dropout", "masked_fill_", "softmax"):
        if action == "compute":
            profiler.profile(opcode)
        latencies = [
            entry["latency_ms"] for key, entry in records.items() if key.startswith(opcode + "__")
        ]
        if not latencies or not all(0 < latency < math.inf for latency in latencies):
            raise RuntimeError(f"Missing or invalid {opcode} measurements for seq={sequence}.")
    print(f"TEST5_ASCEND_PROFILE_READY seq={sequence} entries={len(records)}", flush=True)


if __name__ == "__main__":
    main()

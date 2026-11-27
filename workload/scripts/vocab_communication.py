import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch_npu  # Registers the NPU backend.
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main():
    output = Path(sys.argv[1])
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    shape = (256, 2)
    operations = (dist.ReduceOp.MAX, dist.ReduceOp.SUM, dist.ReduceOp.SUM)
    warmup, samples = 10, 50
    dist.init_process_group("hccl", timeout=timedelta(minutes=3))
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("This vocabulary communication test requires two ranks.")
        group = dist.new_group(ranks=[0, 1], backend="hccl")
        buffers = [torch.empty(shape, dtype=torch.float32, device=device) for _ in operations]
        timings = []
        for index in range(warmup + samples):
            for tensor in buffers:
                tensor.fill_(rank + 1)
            torch.npu.synchronize()
            dist.barrier(group=group)
            torch.npu.synchronize()
            start = perf_counter()
            for tensor, operation in zip(buffers, operations, strict=True):
                dist.all_reduce(tensor, op=operation, group=group)
            torch.npu.synchronize()
            elapsed_ms = (perf_counter() - start) * 1000
            if index >= warmup:
                timings.append(elapsed_ms)

        if not all(torch.all(tensor == value).item() for tensor, value in zip(buffers, (2, 3, 3), strict=True)):
            raise RuntimeError(f"Vocabulary all-reduce check failed on rank {rank}.")
        slower_rank = torch.tensor(timings, dtype=torch.float32, device=device)
        dist.all_reduce(slower_rank, op=dist.ReduceOp.MAX, group=group)
        timings = slower_rank.cpu().tolist()
        if not all(0 < value < math.inf for value in timings):
            raise RuntimeError("Invalid vocabulary communication timings.")
        if rank == 0:
            report = {
                "tp_degree": 2,
                "sequence_length": 256,
                "micro_batch_size": 2,
                "tensor_shape": list(shape),
                "dtype": "float32",
                "reductions": ["max", "sum", "sum"],
                "transport": "native_hccl",
                "warmup_samples": warmup,
                "timed_samples": samples,
                "per_microbatch_ms": mean(timings),
            }
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("TEST5_ASCEND_4L_VOCAB_COMM", json.dumps(report), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

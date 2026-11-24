#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/node-config.sh"

source "$trial_dir/.venv/bin/activate"
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
arena_mlu_require_master
arena_mlu_set_gloo_interface
cd "$trial_dir/arena-mlu-train-smoke/crius-torch"

run_dir="$(mktemp -d "$trial_dir/arena-mlu-dual-benchmark.XXXXXX")"
printf 'Benchmark results directory: %s\n' "$run_dir"

cat > "$run_dir/run-arena-comparison.py" <<'PY'
import inspect
import json
import math
import os
import runpy
import sys
from pathlib import Path
from statistics import mean
from textwrap import dedent
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_mlu
import torch_mlu.utils.gpu_migration
from megatron.core import parallel_state
from torch.distributed.elastic.multiprocessing.errors import record

from crius.train_backend.megatron import transformer_block
from crius.train_backend.megatron.training import trainer, utils

WARMUP = 2
LAYERS = tuple(int(count) for count in os.environ["ARENA_MLU_STAGE_LAYERS"].split(","))
PLAN = os.environ["ARENA_MLU_BENCHMARK_PLAN"]


def apply_pipeline_partition():
    original = transformer_block.TransformerBlock._build_layers
    if len(LAYERS) == 1:
        return

    def current_stage_layers(config):
        if sum(LAYERS) != config.num_layers or len(LAYERS) != config.pipeline_model_parallel_size:
            raise RuntimeError(f"Invalid pipeline partition {LAYERS} for this model.")
        stage = parallel_state.get_pipeline_model_parallel_rank()
        return LAYERS[stage]

    def stage_layer_indices(config, count):
        if getattr(config, "device_free_profile", False):
            return range(count)
        stage = parallel_state.get_pipeline_model_parallel_rank()
        offset = sum(LAYERS[:stage])
        return range(offset, offset + count)

    scope = dict(vars(transformer_block), current_stage_layers=current_stage_layers,
                 stage_layer_indices=stage_layer_indices)
    source = dedent(inspect.getsource(original))
    for old, new in (
        ("self.config.num_layers // global_configs.pipeline_model_parallel_size",
         "current_stage_layers(self.config)"),
        ("range(num_layers)", "stage_layer_indices(self.config, num_layers)"),
    ):
        if source.count(old) != 1:
            raise RuntimeError("Unexpected Crius TransformerBlock implementation.")
        source = source.replace(old, new)
    exec(compile(source, inspect.getfile(original), "exec"), scope)
    partitioned_build = scope["_build_layers"]

    def build_layers(self):
        partitioned_build(self)
        if not getattr(self.config, "device_free_profile", False):
            stage = parallel_state.get_pipeline_model_parallel_rank()
            offset = sum(LAYERS[:stage])
            ids = [layer.layer_number for layer in self.layers]
            expected = list(range(offset + 1, offset + 1 + LAYERS[stage]))
            if ids != expected:
                raise RuntimeError(f"Stage {stage} built layers {ids}, expected {expected}.")
            print(f"PIPELINE_STAGE rank={dist.get_rank()} stage={stage} layers={ids}", flush=True)

    transformer_block.TransformerBlock._build_layers = build_layers


class TimedTrainer(trainer.Trainer):
    def train_step(self):
        dist.barrier()
        start = perf_counter()
        result = super().train_step()
        elapsed = perf_counter() - start
        forward_backward = self.timer._records["train_step"][-1].elapsed
        self.step_timings.append((elapsed, forward_backward))
        return result

    def train(self, metric_types=()):
        self.step_timings = []
        config = self.megatron_config
        start_samples = self.opt_param_scheduler.num_steps
        losses, _ = super().train(metric_types=["train_step"])

        dp_degree = parallel_state.get_data_parallel_world_size()
        global_batch = config.global_batch_size * dp_degree
        updated_samples = self.opt_param_scheduler.num_steps - start_samples
        valid = (
            len(losses) == len(self.step_timings) == config.num_iters
            and updated_samples == config.num_iters * global_batch
            and all(0 < fb <= full < math.inf for full, fb in self.step_timings)
        )
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            valid = valid and all(
                microbatch_losses is not None
                and len(microbatch_losses) == config.num_micro_batches
                and all(math.isfinite(entry["lm loss"].item()) for entry in microbatch_losses)
                for microbatch_losses in losses
            )
        if not valid:
            print(f"Rank {dist.get_rank()} failed the timing, loss, or parameter update check.",
                  file=sys.stderr, flush=True)
        accepted = torch.tensor([float(valid)], dtype=torch.float32, device="mlu:0")
        dist.all_reduce(accepted, op=dist.ReduceOp.MIN)
        if accepted.item() != 1:
            raise RuntimeError("At least one rank produced an invalid benchmark run.")

        durations = torch.tensor(self.step_timings[WARMUP:], dtype=torch.float32, device="mlu:0")
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
        timings = durations.cpu().tolist()
        full_step_s = [full for full, _ in timings]
        forward_backward_s = [fb for _, fb in timings]
        tokens_per_step = global_batch * config.sequence_length
        if dist.get_rank() == 0:
            result = {
                "plan": PLAN,
                "tp_pp_dp": [config.tensor_model_parallel_size, config.pipeline_model_parallel_size, dp_degree],
                "stage_layers": LAYERS,
                "global_batch": global_batch,
                "sequence_length": config.sequence_length,
                "warmup_steps": WARMUP,
                "updated_steps": config.num_iters,
                "step_seconds": full_step_s,
                "forward_backward_seconds": forward_backward_s,
                "tokens_per_second": tokens_per_step / mean(full_step_s),
            }
            Path(os.environ["ARENA_MLU_RESULT_PATH"]).write_text(json.dumps(result, indent=2) + "\n")
            print("RESULT", json.dumps(result), flush=True)
        return losses, mean(full_step_s)


@record
def main():
    apply_pipeline_partition()
    optimizer_factory = utils.get_optimizer_and_scheduler

    def optimizer_with_fixed_loss_scale(*args, **kwargs):
        kwargs["loss_scale"] = 1.0
        return optimizer_factory(*args, **kwargs)

    utils.get_optimizer_and_scheduler = optimizer_with_fixed_loss_scale
    trainer.Trainer = TimedTrainer
    trainer.time = SimpleNamespace(sleep=lambda seconds: None)
    torch.manual_seed(1234)
    try:
        runpy.run_path("scripts/bench_megatron.py", run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

port=29646
run_case() {
    local plan="$1" repetition="$2"
    local tp pp layers
    case "$plan" in
        manual) tp=2; pp=1; layers=4 ;;
        tuned)  tp=1; pp=2; layers=3,1 ;;
    esac

    printf 'Running %s repeat %s with TP/PP/DP=%s/%s/1 and stages %s on %s\n' \
        "$plan" "$repetition" "$tp" "$pp" "$layers" "$host"
    ARENA_MLU_BENCHMARK_PLAN="$plan" ARENA_MLU_STAGE_LAYERS="$layers" \
        ARENA_MLU_RESULT_PATH="$run_dir/$plan-$repetition.json" \
        timeout -k 10s 600s python -m torch.distributed.run \
        --nnodes=2 --nproc-per-node=1 --node-rank="$rank" \
        --master-addr="$ARENA_MLU_MASTER_ADDR" --master-port="$port" --max-restarts=0 \
        "$run_dir/run-arena-comparison.py" \
        --num-nodes 2 --tp-degree "$tp" --pp-degree "$pp" --dp-degree 1 \
        --batch-size 8 --seq-len 256 --num-layers 4 --hidden-size 4096 \
        --num-attention-heads 32 --vocab-size 50000 --num-micro-batches 4 --num-iters 7 \
        2>&1 | tee "$run_dir/$plan-$repetition.log"
    port=$((port + 1))
}

for repetition in 1 2 3; do
    if (( repetition % 2 == 1 )); then
        run_case manual "$repetition"
        run_case tuned "$repetition"
    else
        run_case tuned "$repetition"
        run_case manual "$repetition"
    fi
done

if [[ "$rank" == 0 ]]; then
    python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])
summary = {}
for plan in ("manual", "tuned"):
    runs = [json.loads((root / f"{plan}-{index}.json").read_text()) for index in (1, 2, 3)]
    full = mean(step for run in runs for step in run["step_seconds"])
    forward_backward = mean(step for run in runs for step in run["forward_backward_seconds"])
    tokens_per_step = runs[0]["global_batch"] * runs[0]["sequence_length"]
    summary[plan] = {
        "tp_pp_dp": runs[0]["tp_pp_dp"],
        "stage_layers": runs[0]["stage_layers"],
        "measured_steps": sum(len(run["step_seconds"]) for run in runs),
        "step_ms": full * 1000,
        "forward_backward_ms": forward_backward * 1000,
        "tokens_per_second": tokens_per_step / full,
        "tokens_per_second_by_repeat": [run["tokens_per_second"] for run in runs],
    }
    print(f"SUMMARY {plan}", json.dumps(summary[plan]))
gain = 100 * (summary["tuned"]["tokens_per_second"] / summary["manual"]["tokens_per_second"] - 1)
summary["throughput_change_percent"] = gain
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"SUMMARY throughput_change_percent={gain:.2f}")
PY
fi

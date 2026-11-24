#!/usr/bin/env bash
set -euo pipefail

trial_dir="${1:-${ARENA_MLU_TRIAL_DIR:-$HOME/arena-mlu-work}}"
trial_dir="$(cd "$trial_dir" && pwd)"
bundle_dir="$trial_dir/arena-mlu-train-smoke"
repo_dir="$bundle_dir/crius-torch"
python_bin="$trial_dir/.venv/bin/python"
if [[ ! -x "$python_bin" || ! -f "$repo_dir/scripts/bench_megatron.py" ]]; then
    printf '%s\n' "The MLU virtual environment or benchmark is missing from $trial_dir." >&2
    exit 1
fi

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_dir="$(mktemp -d "$bundle_dir/arena-mlu-matched.XXXXXX")"
printf 'Run directory: %s\n' "$run_dir"

cat > "$run_dir/run-benchmark.py" <<'PY'
import math
import runpy
from statistics import mean
from time import perf_counter
from types import SimpleNamespace

import torch
import torch_mlu
import torch_mlu.utils.gpu_migration
from torch.distributed.elastic.multiprocessing.errors import record

from crius.train_backend.megatron.training import trainer, utils

WARMUP_STEPS = 2


class TimedTrainer(trainer.Trainer):
    def train_step(self):
        start = perf_counter()
        result = super().train_step()
        full_step_seconds = perf_counter() - start
        forward_backward_seconds = self.timer._records["train_step"][-1].elapsed
        self.step_timings.append((forward_backward_seconds, full_step_seconds))
        return result

    def train(self, metric_types=()):
        self.step_timings = []
        config = self.megatron_config
        start_samples = self.opt_param_scheduler.num_steps
        # Retain the base class's timing records while omitting its mislabelled metrics.
        losses, _ = super().train(metric_types=["train_step"])
        if len(losses) != config.num_iters or len(self.step_timings) != config.num_iters:
            raise RuntimeError("The benchmark did not complete the requested training steps.")
        for index, microbatch_losses in enumerate(losses, start=1):
            if len(microbatch_losses) != config.num_micro_batches or any(
                not math.isfinite(entry["lm loss"].item()) for entry in microbatch_losses
            ):
                raise RuntimeError(f"Step {index} did not produce finite losses for each microbatch.")

        # The pinned trainer advances the scheduler only after a successful optimizer step.
        trained_samples = self.opt_param_scheduler.num_steps - start_samples
        if trained_samples != config.global_batch_size * config.num_iters:
            raise RuntimeError(f"Optimizer updates accounted for only {trained_samples} samples.")

        measured = self.step_timings[WARMUP_STEPS:]
        forward_backward_s = mean(timing[0] for timing in measured)
        full_step_s = mean(timing[1] for timing in measured)
        tokens_per_step = config.global_batch_size * config.sequence_length
        if not 0 < forward_backward_s <= full_step_s < math.inf:
            raise RuntimeError("Invalid measured training times.")
        print(f"RESULT warmup_steps={WARMUP_STEPS} measured_steps={len(measured)}")
        print(f"RESULT finite_loss_steps={len(losses)} optimizer_updates={config.num_iters}")
        print(f"RESULT forward_backward_ms={forward_backward_s * 1000:.3f}")
        print(f"RESULT full_train_step_ms={full_step_s * 1000:.3f}")
        print(f"RESULT input_tokens_per_s={tokens_per_step / full_step_s:.2f}", flush=True)
        return losses, full_step_s


@record
def main():
    print(f"Device: {torch.mlu.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__} | torch_mlu: {torch_mlu.__version__}", flush=True)
    create_optimizer = utils.get_optimizer_and_scheduler

    def optimizer_with_fixed_loss_scale(*args, **kwargs):
        kwargs["loss_scale"] = 1.0
        return create_optimizer(*args, **kwargs)

    utils.get_optimizer_and_scheduler = optimizer_with_fixed_loss_scale
    trainer.Trainer = TimedTrainer
    # Only the trainer's artificial delay between iterations uses this module reference.
    trainer.time = SimpleNamespace(sleep=lambda seconds: None)
    runpy.run_path("scripts/bench_megatron.py", run_name="__main__")


if __name__ == "__main__":
    main()
PY

cd "$repo_dir"
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node=1 \
    "$run_dir/run-benchmark.py" \
    --tp-degree 1 --pp-degree 1 --dp-degree 1 \
    --batch-size 8 --seq-len 256 --num-layers 4 \
    --hidden-size 4096 --num-attention-heads 32 --vocab-size 50000 \
    --num-micro-batches 4 --num-iters 7 2>&1 | tee "$run_dir/benchmark.log"

#!/usr/bin/env bash
# Arena test five v6 for two MR-V100 GPUs.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/node-config.sh"

resume=0
tune_only=0
compare_name=
prepare_only=0

usage() {
  printf '用法 %s [prepare 空目录 | {resume,tune,compare} [运行目录]]\n' "${0##*/}" >&2
  exit 2
}

if [[ "${1:-}" == prepare ]]; then
  (( $# == 2 )) || usage
  prepare_only=1
fi

has_saved_run() {
  if [[ "$2" == plan.env ]]; then
    [[ -s "$1/plan.env" && -f "$1/crius-torch/scripts/bench_megatron.py" ]]
    return
  fi
  [[ -s "$1/baseline.json" && -s "$1/baseline.log" && -s "$1/mr-v100-dual-profile.pkl" \
    && -s "$1/$2" && -f "$1/crius-torch/scripts/bench_parallel_tuner.py" ]]
}

if (( $# == 0 || prepare_only )); then
  if [[ ! -r "$arena_mr_source_archive" ]]; then
    printf '找不到源码包 %s，请设置 ARENA_MR_SOURCE_ARCHIVE。\n' "$arena_mr_source_archive" >&2
    exit 1
  fi
  if (( prepare_only )); then
    run_dir=$(cd -- "$2" && pwd -P)
    if [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      printf '源码准备需要空目录，当前为 %s。\n' "$run_dir" >&2
      exit 1
    fi
  else
    mkdir -p -- "$arena_mr_trial_dir"
    run_dir=$(mktemp -d "$arena_mr_trial_dir/arena-iluvatar-test5.XXXXXX")
    run_dir=$(cd -- "$run_dir" && pwd -P)
  fi
  tar -xzf "$arena_mr_source_archive" -C "$run_dir"

  python3 - "$run_dir/crius-torch/scripts/bench_megatron.py" <<'PY_DEVICE'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text(encoding='utf-8')
line = re.compile(r'^([ \t]*)global_configs\.current_device = torch\.device\(device\)[ \t]*$', re.M)
updated, matches = line.subn(r'\1device = "cuda"\n\g<0>', source)
if matches != 1:
    raise SystemExit(f'Expected one training device assignment in {path}, found {matches}')
compile(updated, str(path), 'exec')
path.write_text(updated, encoding='utf-8')
print('Crius training device set to cuda in the extracted source')
PY_DEVICE
elif [[ "$1" == resume || "$1" == tune || "$1" == compare ]] && (( $# <= 2 )); then
  resume=1
  progress_log=p2p.log
  if [[ "$1" == tune ]]; then
    tune_only=1
    progress_log=tuner.log
  elif [[ "$1" == compare ]]; then
    progress_log=plan.env
  fi
  run_dir="${2:-}"
  if [[ -z "$run_dir" ]]; then
    for candidate in "$arena_mr_trial_dir"/arena-iluvatar-test5.*; do
      has_saved_run "$candidate" "$progress_log" || continue
      if [[ -z "$run_dir" || "$candidate/$progress_log" -nt "$run_dir/$progress_log" ]]; then
        run_dir="$candidate"
      fi
    done
  fi
  if [[ -z "$run_dir" ]] || ! has_saved_run "$run_dir" "$progress_log"; then
    printf '找不到可从 %s 继续的测试五目录。\n' "$progress_log" >&2
    exit 1
  fi
  run_dir=$(cd "$run_dir" && pwd -P)
else
  usage
fi

run_log="$run_dir/test5.log"
if [[ "${1:-}" == compare ]]; then
  compare_dir=$(mktemp -d "$run_dir/compare.XXXXXX")
  compare_name=${compare_dir##*/}
  cp -- "$run_dir/plan.env" "$compare_dir/plan.env"
  run_log="$compare_dir/compare.log"
  printf '配对实测输出目录 %s\n' "$compare_dir"
else
  python3 - "$run_dir/crius-torch/crius/loader/model/shadow_loader.py" <<'PY_SHADOW'
from pathlib import Path
import re
from stat import S_IMODE
import sys
from tempfile import TemporaryDirectory

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
call = re.compile(r"^([ \t]*)model\.to\(self\.device_str\)[ \t]*$", re.M)
replacement = r"\1_arena_move_shadow_model(model, self.device_str)"
if "def _arena_move_shadow_model(" not in source:
    updated, matches = call.subn(replacement, source)
    if matches != 1:
        raise SystemExit(f"Expected one shadow model device conversion in {path}, found {matches}")
    updated = updated.rstrip() + r'''


def _arena_move_shadow_model(model, device):
    import torch

    # Stage the conversions before rebinding so tied parameters stay tied.
    slots = [
        (registry, name, tensor)
        for module in model.modules()
        for registry in (module._parameters, module._buffers)
        for name, tensor in registry.items()
        if tensor is not None
    ]
    replacements = {}
    with torch.no_grad():
        for _, _, tensor in slots:
            key = id(tensor)
            if key in replacements:
                continue
            moved = tensor.to(device=device)
            if moved is not tensor and isinstance(tensor, torch.nn.Parameter):
                moved = torch.nn.Parameter(moved, requires_grad=tensor.requires_grad)
                if tensor.grad is not None:
                    moved.grad = tensor.grad.to(device=device)
                for name, value in vars(tensor).items():
                    vars(moved).setdefault(name, value)
            replacements[key] = moved
    for registry, name, tensor in slots:
        registry[name] = replacements[id(tensor)]
    print(f"ARENA_SHADOW_MOVE device={device} unique_tensors={len(replacements)}", flush=True)
    return model
'''
    compile(updated, str(path), "exec")
    with TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(updated, encoding="utf-8")
        staged.chmod(S_IMODE(path.stat().st_mode))
        staged.replace(path)
    print(f"Applied shadow model device conversion in {path}")
PY_SHADOW
fi

arena_mr_docker_options
printf '测试五输出目录 %s\n' "$run_dir"

docker run "${docker_options[@]}" -e "ARENA_TEST5_RESUME=$resume" \
  -e "ARENA_TEST5_TUNE_ONLY=$tune_only" -e "ARENA_TEST5_COMPARE=$compare_name" \
  -e "ARENA_MR_PREPARE_ONLY=$prepare_only" \
  --mount "type=bind,source=$run_dir,target=/workspace/arena" \
  -w /workspace/arena/crius-torch \
  "$arena_mr_image" bash -s <<'ARENA_CONTAINER' 2>&1 | tee -a "$run_log"
set -euo pipefail
export LD_LIBRARY_PATH="/opt/arena-corex-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/workspace/arena/crius-torch:/workspace/arena/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

python3 -m pip install --disable-pip-version-check --timeout 20 --retries 1 --only-binary=:all: --no-deps --target /tmp/arena-python sentencepiece==0.2.1 pybind11==2.13.6
export PYTHONPATH="/tmp/arena-python:$PYTHONPATH"

if [[ "$ARENA_TEST5_RESUME" == 0 ]]; then
  dataset_dir=/workspace/arena/Megatron-LM/megatron/core/datasets
  python_include=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
  build_packages=()
  command -v make >/dev/null || build_packages+=(make)
  command -v g++ >/dev/null || build_packages+=(g++)
  [[ -r "$python_include/Python.h" ]] || build_packages+=(python3.12-dev)
  if ((${#build_packages[@]})); then
    apt-get update -qq
    apt-get install -y --no-install-recommends "${build_packages[@]}"
  fi

  helper_includes=$(python3 -m pybind11 --includes)
  if [[ ! -r "$python_include/Python.h" ]]; then
    helper_includes+=" -I/usr/include/python3.12"
  fi
  helper_suffix=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
  make -C "$dataset_dir" "LIBEXT=$helper_suffix" "CPPFLAGS=$helper_includes"
  python3 -u -c 'from megatron.core.datasets import helpers; print(f"DATASET_HELPERS_READY {helpers.__file__}")'
fi

if [[ "$ARENA_MR_PREPARE_ONLY" == 1 ]]; then
  exit 0
fi

if [[ "$ARENA_TEST5_RESUME" == 0 || -n "$ARENA_TEST5_COMPARE" ]]; then
  cat > /tmp/arena_test5_baseline.py <<'PY_BASELINE'
import json
import math
import os
import runpy
from pathlib import Path
from statistics import mean
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from crius.train_backend.megatron.training import trainer, utils
from megatron.core import parallel_state

WARMUP_STEPS = 2


class TimedTrainer(trainer.Trainer):
    def train_step(self):
        dist.barrier()
        torch.cuda.synchronize()
        start = perf_counter()
        result = super().train_step()
        torch.cuda.synchronize()
        full_step_s = perf_counter() - start
        forward_backward_s = self.timer._records["train_step"][-1].elapsed
        self.step_timings.append((full_step_s, forward_backward_s))
        return result

    def train(self, metric_types=()):
        config = self.megatron_config
        self.step_timings = []
        start_samples = self.opt_param_scheduler.num_steps
        losses, _ = super().train(metric_types=["train_step"])
        updated_samples = self.opt_param_scheduler.num_steps - start_samples
        degrees = [
            parallel_state.get_tensor_model_parallel_world_size(),
            parallel_state.get_pipeline_model_parallel_world_size(),
            parallel_state.get_data_parallel_world_size(),
        ]
        valid = (
            torch.cuda.current_device() == int(os.environ["LOCAL_RANK"])
            and degrees == [2, 1, 1]
            and config.num_layers == 4
            and config.global_batch_size == 8
            and config.sequence_length == 256
            and config.num_micro_batches == 4
            and config.num_iters == 7
            and len(losses) == len(self.step_timings) == config.num_iters
            and updated_samples == config.global_batch_size * config.num_iters
            and all(0 < fb <= full < math.inf for full, fb in self.step_timings)
            and all(
                step is not None
                and len(step) == config.num_micro_batches
                and all(math.isfinite(entry["lm loss"].item()) for entry in step)
                for step in losses
            )
        )
        device = torch.device("cuda", torch.cuda.current_device())
        if not valid:
            print(f"Rank {dist.get_rank()} failed a layout, timing, loss, or optimizer update check.", flush=True)
        accepted = torch.tensor(float(valid), device=device)
        dist.all_reduce(accepted, op=dist.ReduceOp.SUM)
        if accepted.item() != dist.get_world_size():
            raise RuntimeError("At least one rank produced an invalid training run.")

        durations = torch.tensor(self.step_timings[WARMUP_STEPS:], dtype=torch.float32, device=device)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
        timings = durations.cpu().tolist()
        full_seconds = [full for full, _ in timings]
        forward_backward_seconds = [fb for _, fb in timings]
        full_mean = mean(full_seconds)
        if dist.get_rank() == 0:
            result = {
                "tp_pp_dp": degrees,
                "stage_layers": [config.num_layers],
                "global_batch": config.global_batch_size,
                "sequence_length": config.sequence_length,
                "warmup_steps": WARMUP_STEPS,
                "optimizer_updates": config.num_iters,
                "timed_steps": len(full_seconds),
                "step_seconds": full_seconds,
                "forward_backward_seconds": forward_backward_seconds,
                "step_ms": 1000 * full_mean,
                "forward_backward_ms": 1000 * mean(forward_backward_seconds),
                "tokens_per_second": config.global_batch_size * config.sequence_length / full_mean,
            }
            output = Path(os.environ["ARENA_TEST5_RESULT_PATH"])
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"TEST5_{output.stem.upper()}", json.dumps(result), flush=True)
        return losses, full_mean


@record
def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    create_optimizer = utils.get_optimizer_and_scheduler

    def optimizer_with_fixed_loss_scale(*args, **kwargs):
        kwargs["loss_scale"] = 1.0
        return create_optimizer(*args, **kwargs)

    utils.get_optimizer_and_scheduler = optimizer_with_fixed_loss_scale
    trainer.Trainer = TimedTrainer
    trainer.time = SimpleNamespace(sleep=lambda _: None)
    torch.manual_seed(1234)
    try:
        runpy.run_path("scripts/bench_megatron.py", run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY_BASELINE

  run_timed_training() {
    local result=$1
    ARENA_TEST5_RESULT_PATH="$result" \
      timeout -k 10s 600s python3 -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=2 /tmp/arena_test5_baseline.py \
        --num-nodes 1 --tp-degree "$2" --pp-degree "$3" --dp-degree "$4" \
        --batch-size 8 --seq-len 256 --num-layers 4 --hidden-size 4096 \
        --num-attention-heads 32 --vocab-size 50000 --num-micro-batches 4 --num-iters 7 \
        2>&1 | tee "${result%.json}.log"
  }

  if [[ -z "$ARENA_TEST5_COMPARE" ]]; then
    run_timed_training /workspace/arena/baseline.json 2 1 1
  else
    compare_path="/workspace/arena/$ARENA_TEST5_COMPARE"
    tuned_degrees=$(python3 - "$compare_path/plan.env" <<'PY_PLAN'
from pathlib import Path
import re
import sys

plan = Path(sys.argv[1])
degrees = {}
for line in plan.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    match = re.fullmatch(r"export CRIUS_TUNED_(TP|PP|DP)_DEGREE=([1-9][0-9]*)", line)
    if match is None or match[1] in degrees:
        raise SystemExit(f"无法读取 {plan} 中的选案参数 {line!r}")
    degrees[match[1]] = int(match[2])
if degrees != {"TP": 2, "PP": 1, "DP": 1}:
    raise SystemExit(f"本次配对复测需要 TP/PP/DP=2/1/1 的选案，读到 {degrees}")
print(degrees["TP"], degrees["PP"], degrees["DP"])
PY_PLAN
    )
    read -r tp pp dp <<< "$tuned_degrees"
    printf 'ARENA_TEST5_V6_COMPARE TP/PP/DP=%s/%s/%s\n' "$tp" "$pp" "$dp"
    run_timed_training "$compare_path/manual.json" 2 1 1
    run_timed_training "$compare_path/arena.json" "$tp" "$pp" "$dp"
    python3 - "$compare_path" <<'PY_COMPARE'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
manual = json.loads((path / "manual.json").read_text(encoding="utf-8"))
arena = json.loads((path / "arena.json").read_text(encoding="utf-8"))
plan_keys = ("tp_pp_dp", "stage_layers", "global_batch", "sequence_length", "warmup_steps", "timed_steps")
plan = {key: manual[key] for key in plan_keys}
if plan != {key: arena[key] for key in plan_keys}:
    raise SystemExit("两组训练的并行布局、负载或计时设置不同。")
metrics = ("step_ms", "forward_backward_ms", "tokens_per_second")
summary = {
    **plan,
    "manual": {key: manual[key] for key in metrics},
    "arena": {key: arena[key] for key in metrics},
}
(path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("TEST5_COMPARE", json.dumps(summary), flush=True)
PY_COMPARE
    exit 0
  fi
fi

profile=/workspace/arena/mr-v100-dual-profile.pkl
if [[ "$ARENA_TEST5_TUNE_ONLY" == 0 ]]; then
  cat > /tmp/arena_test5_profile.py <<'PY_PROFILE'
import math
import os
import sys
from pathlib import Path

import torch
from crius.offline.offline_profiler import CriusOfflineDataManager, OfflineProfileConfigs
from torch.distributed.elastic.multiprocessing.errors import record

MICRO_BATCH = 2
SEQUENCE = 256
HIDDEN = 4096
ACTIVATION_ELEMENTS = MICRO_BATCH * SEQUENCE * HIDDEN
ALL_REDUCE_ELEMENTS = (2, 512, 65_536, ACTIVATION_ELEMENTS, 2**24, 2**26, HIDDEN * 50_000)


@record
def main():
    action, path = sys.argv[1:]
    profile = Path(path)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    heads = [16, 32]
    hidden_dims = [2048, HIDDEN, 8192, 16384]
    configs = OfflineProfileConfigs(
        device=f"cuda:{local_rank}",
        dtype=torch.float16,
        min_batch_size=MICRO_BATCH,
        max_batch_size=MICRO_BATCH,
        min_seq_len=SEQUENCE,
        max_seq_len=SEQUENCE,
        num_warmup_steps=3,
        num_profile_steps=7,
        force_overwrite=(action != "compute"),
        use_torch_empty_to_init_weight=False,
        use_many_to_many_p2p_offline_data=True,
        many_to_many_p2p_rank_mapping={0: 1, 1: 0},
        many_to_many_sender_ranks=[0],
        many_to_many_recver_ranks=[1],
        comm_data_sizes=list(ALL_REDUCE_ELEMENTS if action == "all_reduce" else [ACTIVATION_ELEMENTS]),
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
        profiler.profile(action)
        return

    records = profiler.recorder.data
    required = [f"all_reduce__n1__gs2__d{n}" for n in ALL_REDUCE_ELEMENTS]
    for key in required:
        latency = records.get(key, {}).get("latency_ms")
        if not isinstance(latency, (float, int)) or not 0 < latency < math.inf:
            raise SystemExit(f"Missing or invalid single-node two-GPU profile for {key}. Check the communication profiling logs.")
    p2p_key = f"p2p__n1__gs2__d{ACTIVATION_ELEMENTS}"
    latency = records.get(p2p_key, {}).get("latency_ms")
    if not isinstance(latency, list) or len(latency) != 3 or any(
        not 0 < value < math.inf for value in latency
    ):
        raise SystemExit(f"Missing or invalid single-node two-GPU profile for {p2p_key}. Check the communication profiling logs.")

    for opcode in ("gemm", "bmm", "baddbmm", "layernorm", "dropout", "masked_fill_", "softmax"):
        profiler.profile(opcode)


if __name__ == "__main__":
    main()
PY_PROFILE

  if [[ "$ARENA_TEST5_RESUME" == 0 ]]; then
    for action in all_reduce p2p; do
      printf 'Profiling %s on both MR-V100 GPUs\n' "$action"
      timeout -k 10s 600s python3 -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=2 /tmp/arena_test5_profile.py \
        "$action" "$profile" 2>&1 | tee "/workspace/arena/$action.log"
    done
  else
    printf '复用已保存的基线和通信数据，从算子采样继续。\n'
  fi
  timeout -k 10s 600s python3 -u /tmp/arena_test5_profile.py compute "$profile" \
    2>&1 | tee -a /workspace/arena/compute.log
else
  printf '复用已保存的基线、通信和算子数据，从 Arena 选案继续。\n'
fi

cat > /tmp/arena_test5_tune.py <<'PY_TUNE'
import re
import sys
from pathlib import Path

import torch
from crius.global_envs import global_configs
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main():
    torch.cuda.set_device(0)
    global_configs.per_device_memory_capacity_in_gb = min(
        torch.cuda.get_device_properties(index).total_memory for index in range(2)
    ) / 1024**3
    entry = Path("scripts/bench_parallel_tuner.py").resolve()
    source = entry.read_text(encoding="utf-8")
    device_line = re.compile(
        r'^([ \t]*)global_configs\.current_device = torch\.device\(device\)[ \t]*$', re.M
    )
    source, matches = device_line.subn(r'\1device = "cuda"\n\g<0>', source)
    if matches != 1:
        raise RuntimeError(f"Expected one device assignment in {entry}, found {matches}")
    print("Tuner topology: one node with two MR-V100 GPUs", flush=True)
    sys.argv[0] = str(entry)
    # Crius treats one LayerNorm module call as one pipeline boundary.
    # Preserve those calls so weight and bias reads do not count as extra boundaries.
    with torch._dynamo.config.patch(inline_inbuilt_nn_modules=False):
        print("ARENA_TEST5_V6_TUNER preserving LayerNorm module calls in the exported graph", flush=True)
        exec(compile(source, str(entry), "exec"), {"__name__": "__main__", "__file__": str(entry)})


if __name__ == "__main__":
    main()
PY_TUNE

plan=/workspace/arena/plan.env
timeout -k 10s 600s python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=1 /tmp/arena_test5_tune.py \
  --num-mlus 2 --batch-size 8 --seq-len 256 --num-layers 4 --hidden-size 4096 \
  --num-attention-heads 32 --vocab-size 50000 --num-micro-batches 4 \
  --profiled-data-path "$profile" --env-output "$plan" 2>&1 | tee -a /workspace/arena/tuner.log

test -s "$plan"
printf '\n'
grep -m 1 '^TEST5_BASELINE ' /workspace/arena/baseline.log
printf '\nTEST5_PLAN\n'
cat "$plan"
ARENA_CONTAINER

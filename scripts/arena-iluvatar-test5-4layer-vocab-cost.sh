#!/usr/bin/env bash
# Compare the four-layer TP=2 and PP=2 candidates on two MR-V100 GPUs.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/node-config.sh"

case $# in
  1) run_dir=$(cd -- "$1" && pwd -P) ;;
  0)
    compare_dir=
    for summary in "$arena_mr_trial_dir"/arena-iluvatar-test5-vocab.*/compare.*/summary.json; do
      [[ -s "$summary" && -f "${summary%/*}/../crius-torch/arena_mr_vocab_check.py" ]] || continue
      if [[ -z "$compare_dir" || "$summary" -nt "$compare_dir/summary.json" ]]; then
        compare_dir=${summary%/*}
      fi
    done
    if [[ -z "$compare_dir" ]]; then
      printf '找不到词表修复目录。请传入已准备的目录，或运行 scripts/run-test5.sh。\n' >&2
      exit 1
    fi
    run_dir=$(cd -- "$compare_dir/.." && pwd -P)
    ;;
  *) printf '用法 %s [词表修复目录]\n' "${0##*/}" >&2; exit 2 ;;
esac

if [[ ! -f "$run_dir/crius-torch/arena_mr_vocab_check.py" ]]; then
  printf '目录 %s 缺少词表修复。请先运行 arena-iluvatar-test5-vocab-fix.sh prepare。\n' "$run_dir" >&2
  exit 1
fi
test -f "$run_dir/crius-torch/scripts/bench_parallel_tuner.py"
test -f "$run_dir/crius-torch/scripts/bench_megatron.py"
mkdir -p -- "$arena_mr_trial_dir"
output=$(mktemp -d "$arena_mr_trial_dir/arena-iluvatar-test5-4layer-vocab-cost.XXXXXX")
output=$(cd -- "$output" && pwd -P)
printf 'TEST5_MR_4L_DIR %s\n' "$output"

cached_profile="${ARENA_MR_PROFILE_FILE:-}"
if [[ -n "$cached_profile" ]]; then
  if [[ ! -s "$cached_profile" ]]; then
    printf '找不到可用的采样文件 %s。\n' "$cached_profile" >&2
    exit 1
  fi
  cached_profile=$(realpath -- "$cached_profile")
elif (( $# == 0 )); then
  cached_profile=$(python3 - "$arena_mr_trial_dir" <<'PY_FIND_PROFILE'
import json
import sys
from pathlib import Path


home = Path(sys.argv[1])
profiles = sorted(
    home.glob("arena-iluvatar-test5.*/mr-v100-dual-profile.pkl"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
for profile in profiles:
    try:
        baseline = json.loads((profile.parent / "baseline.json").read_text(encoding="utf-8"))
        if (
            profile.stat().st_size
            and (baseline["sequence_length"], baseline["global_batch"], baseline["stage_layers"]) == (256, 8, [4])
            and baseline["tp_pp_dp"] == [2, 1, 1]
        ):
            print(profile)
            break
    except (OSError, ValueError, KeyError):
        continue
PY_FIND_PROFILE
  )
fi
if [[ -n "$cached_profile" ]]; then
  cp -- "$cached_profile" "$output/profile-v6.pkl"
  printf '%s\n' "$cached_profile" > "$output/profile-source.txt"
fi

arena_mr_docker_options

docker run "${docker_options[@]}" \
  --mount "type=bind,source=$run_dir,target=/workspace/arena,readonly" \
  --mount "type=bind,source=$output,target=/workspace/test5-results" \
  -w /workspace/arena/crius-torch \
  "$arena_mr_image" bash -s <<'CONTAINER'
set -euo pipefail
export LD_LIBRARY_PATH="/opt/arena-corex-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/tmp/arena-python:/workspace/arena/crius-torch:/workspace/arena/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python3 -m pip install --disable-pip-version-check --timeout 20 --retries 1 \
  --only-binary=:all: --no-deps --target /tmp/arena-python sentencepiece==0.2.1

cat > /tmp/arena_mr_vocab_profile.py <<'PY_PROFILE'
import math
import os
import sys
from pathlib import Path

import torch
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
    torch.cuda.set_device(local_rank)
    heads = [16, 32]
    hidden_dims = [2048, HIDDEN, 8192, 16384]
    configs = OfflineProfileConfigs(
        device=f"cuda:{local_rank}",
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
            raise SystemExit(f"Missing or invalid single-node two-GPU profile for {key}. Check the communication profiling logs.")
    p2p_key = f"p2p__n1__gs2__d{activation}"
    latency = records.get(p2p_key, {}).get("latency_ms")
    if not isinstance(latency, list) or len(latency) != 3 or any(
        not 0 < value < math.inf for value in latency
    ):
        raise SystemExit(f"Missing or invalid single-node two-GPU profile for {p2p_key}. Check the communication profiling logs.")

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
    print(f"TEST5_MR_PROFILE_READY seq={sequence} entries={len(records)}", flush=True)


if __name__ == "__main__":
    main()
PY_PROFILE

cat > /tmp/arena_mr_vocab_measure.py <<'PY_MEASURE'
import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main():
    output = Path(sys.argv[1])
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    shape = (256, 2)
    operations = (dist.ReduceOp.MAX, dist.ReduceOp.SUM, dist.ReduceOp.SUM)
    warmup, samples = 10, 50
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("This vocabulary communication test requires two ranks.")
        group = dist.new_group(ranks=[0, 1], backend="nccl")
        buffers = [torch.empty(shape, dtype=torch.float32, device=device) for _ in operations]
        timings = []
        for index in range(warmup + samples):
            for tensor in buffers:
                tensor.fill_(rank + 1)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            start = perf_counter()
            for tensor, operation in zip(buffers, operations, strict=True):
                dist.all_reduce(tensor, op=operation, group=group)
            torch.cuda.synchronize()
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
                "transport": "native_nccl",
                "warmup_samples": warmup,
                "timed_samples": samples,
                "per_microbatch_ms": mean(timings),
            }
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("TEST5_MR_4L_VOCAB_COMM", json.dumps(report), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY_MEASURE

cat > /tmp/arena_mr_vocab_tune.py <<'PY_TUNE'
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from crius.compile import postcompile
from crius.core.grid_planner.planner import CriusParallelPlanner
from crius.core.profiler import CriusDistProfiler
from crius.core.tuner import CriusParallelTuner
from crius.global_envs import global_configs
from crius.loader.model import shadow_loader
from crius.train_backend.megatron.gpt_model import GPTModel
from torch.distributed.elastic.multiprocessing.errors import record


def virtual_vocab_loss(self, labels, logits):
    # The graph compiler uses a one-rank virtual group and unsharded logits.
    return F.cross_entropy(logits.float().permute(1, 2, 0), labels, reduction="none")


def move_shadow_model(model, device):
    # Match the v6 loader's parameter conversion, including any tied parameters.
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
    return model


def shadow_load():
    original = shadow_loader.ShadowModelLoader.load
    if hasattr(shadow_loader, "_arena_move_shadow_model"):
        return original
    source = dedent(inspect.getsource(original))
    original_call = "model.to(self.device_str)"
    if source.count(original_call) != 1:
        raise RuntimeError("Cannot identify the shadow loader's device conversion.")
    source = source.replace(original_call, "move_shadow_model(model, self.device_str)", 1)
    namespace = dict(vars(shadow_loader), move_shadow_model=move_shadow_model)
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    return namespace["load"]


def layout_key(plan):
    return (
        (plan.tensor_parallel_degree, plan.pipeline_parallel_degree, plan.data_parallel_degree),
        tuple(int(count) for count in plan.pipeline_num_layers_per_stage),
    )


def include_even_partitions(planner, grid, precompiled_graph, generate_native):
    # Keep the native split in each PP grid and add the even split when it differs.
    # The installed training adapter replays TP and PP with DP=1.
    native = generate_native(planner, grid, precompiled_graph)
    candidates = [plan for plan in native if plan.data_parallel_degree == 1]
    entry = next(
        plan for plan in planner._enumerate_candidate_parallelism_plans(grid)
        if plan.data_parallel_degree == 1
    )
    pp = entry.pipeline_parallel_degree
    if entry.num_layers % pp:
        raise RuntimeError("This test requires equal-sized transformer-layer groups.")
    equal_layers = (entry.num_layers // pp,) * pp
    key = ((entry.tensor_parallel_degree, pp, 1), equal_layers)
    if any(layout_key(plan) == key for plan in candidates):
        return candidates

    entry.pipeline_stage_partition_method = "uniform"
    graph, backward = postcompile(precompiled_graph, entry)
    planner.partition_pipeline_stages(graph, entry, backward, grid)
    markers = entry.pipeline_boundary_start_marker
    observed = [
        node.stage_index for node in graph.nodes
        if node.op == "call_module" and "L__self___module" in str(node.target)
        and any(marker in str(node.target) for marker in markers)
    ]
    expected = [stage for stage, count in enumerate(equal_layers) for _ in range(count)]
    if layout_key(entry) != key or observed != expected:
        raise RuntimeError(f"Unexpected transformer-layer boundaries: {observed}; expected {expected}.")
    entry.graph, entry.bw_graph = graph, backward
    return [*candidates, entry]


def corrected_pipeline_latency(estimate, profiler, stages, grad_sync, p2p, plan, only_forward, vocab_ms):
    original = estimate(profiler, stages, grad_sync, p2p, plan, only_forward)
    if plan.tensor_parallel_degree == 1:
        return original, original
    if plan.tensor_parallel_degree != 2:
        raise RuntimeError(f"Missing vocabulary communication measurements for {layout_key(plan)}.")

    # The 1F1B latency formula already multiplies per-stage cost by the microbatch count.
    adjusted = [list(stage) for stage in stages]
    adjusted[-1][0] += vocab_ms
    return original, estimate(profiler, adjusted, grad_sync, p2p, plan, only_forward)


def name_layout(degrees, layers):
    if (degrees, layers) == ((2, 1, 1), (4,)):
        return "tp2"
    if degrees == (1, 2, 1) and len(layers) == 2 and min(layers) > 0 and sum(layers) == 4:
        return "pp2_uniform" if layers == (2, 2) else "pp2_auto"
    raise RuntimeError(f"Invalid four-layer layout {degrees}, stage layers {layers}.")


def finite_latency(latency):
    return float(latency) if latency is not None and 0 < latency < math.inf else None


@record
def main():
    torch.cuda.set_device(0)
    global_configs.per_device_memory_capacity_in_gb = min(
        torch.cuda.get_device_properties(i).total_memory for i in range(2)
    ) / 1024**3

    output = Path(os.environ["ARENA_MR_VOCAB_OUTPUT"])
    sample = json.loads((output / "vocab-communication.json").read_text(encoding="utf-8"))
    vocab_ms = finite_latency(sample["per_microbatch_ms"])
    if (
        (sample["tp_degree"], sample["tensor_shape"], sample["dtype"]) != (2, [256, 2], "float32")
        or sample["reductions"] != ["max", "sum", "sum"]
        or sample["transport"] != "native_nccl"
        or (sample["warmup_samples"], sample["timed_samples"]) != (10, 50)
        or vocab_ms is None
    ):
        raise RuntimeError(f"Vocabulary communication sample does not match the comparison: {sample}.")

    entry = Path("scripts/bench_parallel_tuner.py").resolve()
    source = entry.read_text(encoding="utf-8")
    device_line = re.compile(
        r'^([ \t]*)global_configs\.current_device = torch\.device\(device\)[ \t]*$', re.M
    )
    source, matches = device_line.subn(r'\1device = "cuda"\n\g<0>', source)
    if matches != 1:
        raise RuntimeError(f"Expected one device assignment in {entry}, found {matches}.")

    native_planner = CriusParallelPlanner.generate_pareto_optimal_plans
    tune = CriusParallelTuner.tune
    estimate = CriusDistProfiler.profile
    estimate_pipeline = CriusDistProfiler._model_e2e_pipeline_latency_with_oom_check
    scored = {}
    before_vocab = {}

    def expand_grid(self, grid, precompiled_graph):
        return include_even_partitions(self, grid, precompiled_graph, native_planner)

    def model_with_vocab_cost(self, stages, grad_sync, p2p, parallelism_plan, only_forward=False):
        original, adjusted = corrected_pipeline_latency(
            estimate_pipeline, self, stages, grad_sync, p2p, parallelism_plan, only_forward, vocab_ms
        )
        before_vocab[layout_key(parallelism_plan)] = finite_latency(original)
        return adjusted

    def score_candidate(self, *args, **kwargs):
        kwargs["check_oom"] = True
        latency = finite_latency(estimate(self, *args, **kwargs))
        plan = kwargs["parallelism_plan"]
        degrees, layers = key = layout_key(plan)
        unadjusted = before_vocab.get(key)
        added = None
        if latency is not None:
            if unadjusted is None or plan.num_microbatches != 4:
                raise RuntimeError(f"Missing unadjusted four-microbatch estimate for {key}.")
            added = latency - unadjusted
            expected = 4 * vocab_ms if degrees[0] == 2 else 0.0
            if not math.isclose(added, expected, rel_tol=1e-7, abs_tol=1e-7):
                raise RuntimeError(f"Unexpected vocabulary cost for {key}: {added}, expected {expected}.")
        proposal = {
            "case": name_layout(degrees, layers),
            "tp_pp_dp": list(degrees),
            "stage_layers": list(layers),
            "partition_method": plan.pipeline_stage_partition_method,
            "estimated_before_vocab_ms": unadjusted,
            "estimated_forward_backward_ms": latency,
            "vocab_added_to_estimate_ms": added,
        }
        scored[key] = proposal
        print("TEST5_MR_4L_VOCAB_CANDIDATE", json.dumps(proposal), flush=True)
        return latency if latency is not None else math.inf

    def record_selection(self, *args, **kwargs):
        selected, latency = tune(self, *args, **kwargs)
        candidate_path = output / "candidates.json"
        candidate_path.write_text(json.dumps(list(scored.values()), indent=2) + "\n", encoding="utf-8")
        by_case = {proposal["case"]: proposal for proposal in scored.values()}
        if not {"tp2", "pp2_uniform"}.issubset(by_case):
            raise RuntimeError(f"The TP=2 and uniform PP=2 candidates did not both reach the scorer. See {candidate_path}.")
        if selected is None or finite_latency(latency) is None:
            raise RuntimeError("The Arena tuner did not return a valid candidate.")

        valid = [proposal for proposal in scored.values() if proposal["estimated_forward_backward_ms"] is not None]
        unadjusted_winner = min(valid, key=lambda proposal: proposal["estimated_before_vocab_ms"])
        selected_score = scored[layout_key(selected)]
        selection = {"before_vocab": unadjusted_winner, "after_vocab": selected_score}
        (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        print("TEST5_MR_4L_VOCAB_SELECTED", json.dumps(selection), flush=True)
        return selected, latency

    sys.argv[0] = str(entry)
    try:
        with (
            torch._dynamo.config.patch(inline_inbuilt_nn_modules=False),
            patch.object(GPTModel, "compute_language_model_loss", virtual_vocab_loss),
            patch.object(shadow_loader.ShadowModelLoader, "load", shadow_load()),
            patch.object(CriusParallelPlanner, "generate_pareto_optimal_plans", expand_grid),
            patch.object(CriusParallelTuner, "tune", record_selection),
            patch.object(CriusDistProfiler, "profile", score_candidate),
            patch.object(CriusDistProfiler, "_model_e2e_pipeline_latency_with_oom_check", model_with_vocab_cost),
        ):
            exec(compile(source, str(entry), "exec"), {"__name__": "__main__", "__file__": str(entry)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY_TUNE

cat > /tmp/arena_mr_vocab_train.py <<'PY_TRAIN'
import inspect
import json
import math
import os
import runpy
import sys
from contextlib import ExitStack
from pathlib import Path
from statistics import mean
from textwrap import dedent
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

import arena_mr_vocab_check
from crius.train_backend.megatron import layers, transformer_block
from crius.train_backend.megatron.training import trainer, utils
from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.optimizer.grad_scaler import ConstantGradScaler
from megatron.core.utils import get_attr_wrapped_model

WARMUP = 2
TIMED = 5
BATCH = 8
MICROBATCHES = 4
LAYERS = 4


def install_unit_loss_scale(stack):
    create_optimizer = utils.get_optimizer_and_scheduler
    native_inverse = ConstantGradScaler.inv_scale

    def create_unit_scale_optimizer(*args, **kwargs):
        kwargs["loss_scale"] = 1.0
        optimizer, scheduler = create_optimizer(*args, **kwargs)
        scaler = optimizer.grad_scaler
        if not isinstance(scaler, ConstantGradScaler) or scaler.scale.item() != 1.0:
            raise RuntimeError("Expected a constant loss scale of 1.0 for the MR-V100 replay.")
        unit_inverse = torch.ones_like(scaler.scale)

        def inv_scale(current_scaler):
            if current_scaler is scaler:
                return unit_inverse
            return native_inverse.fget(current_scaler)

        # This scaler is fixed at 1.0. Its exact inverse needs no FP64 operations.
        stack.enter_context(patch.object(ConstantGradScaler, "inv_scale", property(inv_scale)))
        return optimizer, scheduler

    stack.enter_context(patch.object(utils, "get_optimizer_and_scheduler", create_unit_scale_optimizer))


class LossCheck:
    def __init__(self, model, group):
        self.compute = model.compute_language_model_loss
        self.group = group
        self.forward_checked = False
        self.backward_checked = False

    def __call__(self, labels, logits):
        loss = self.compute(labels, logits)
        if self.forward_checked:
            return loss

        # Sample tokens across the first microbatch without gathering the entire FP32 vocabulary.
        vocab_per_rank = logits.size(-1)
        stride = max(1, labels.numel() // 128)
        detached = logits.detach().flatten(0, 1)[::stride].contiguous()
        if self.group.size() == 1:
            full = detached.float()
        else:
            gathered = [torch.empty_like(detached) for _ in range(self.group.size())]
            dist.all_gather(gathered, detached, group=self.group)
            full = torch.cat(gathered, dim=-1).float()
        targets = labels.transpose(0, 1).reshape(-1)[::stride].contiguous()
        expected = F.cross_entropy(full, targets, reduction="none")
        actual = loss.detach().transpose(0, 1).reshape(-1)[::stride]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-5)
        self.loss_error = (actual - expected).abs().max().item()

        probabilities = full.softmax(dim=-1)
        probabilities[torch.arange(targets.numel(), device=targets.device), targets] -= 1
        start = self.group.rank() * vocab_per_rank
        partial = probabilities[:, start:start + vocab_per_rank].contiguous()

        def record_upstream(grad_loss):
            self.upstream = grad_loss.detach().transpose(0, 1).reshape(-1)[::stride].unsqueeze(-1)

        def check_gradient(grad_logits):
            actual_grad = grad_logits.detach().flatten(0, 1)[::stride]
            expected_grad = (partial * self.upstream).to(actual_grad.dtype)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-3, atol=3e-7)
            self.gradient_error = (actual_grad.float() - expected_grad.float()).abs().max().item()
            self.backward_checked = True

        loss.register_hook(record_upstream)
        logits.register_hook(check_gradient)
        self.forward_checked = True
        return loss


class WeightSamples:
    def __init__(self, optimizer, model):
        sample_names = ["decoder.layers.0.input_layernorm.weight"]
        if model.post_process:
            sample_names.append("output_layer.weight")
        else:
            sample_names.append("decoder.layers.0.mlp.linear_fc1.weight")
        masters = {
            id(weight): master
            for fp16_group, fp32_group in zip(
                optimizer.float16_groups, optimizer.fp32_from_float16_groups, strict=True
            )
            for weight, master in zip(fp16_group, fp32_group, strict=True)
        }
        self.weights = {}
        for name in sample_names:
            weight = model.get_parameter(name)
            master = masters.get(id(weight))
            if weight.dtype != torch.float16 or master is None or master.dtype != torch.float32:
                raise RuntimeError(f"Missing FP16/FP32 pair for {name}.")
            self.weights[name] = weight, master

    @staticmethod
    def sample(tensor):
        values = tensor.detach()
        if values.ndim == 2:
            rows, columns = values.shape
            values = values[::max(1, rows // 64), ::max(1, columns // 64)]
        return values.clone(memory_format=torch.contiguous_format)

    def snapshot(self):
        return tuple((self.sample(weight), self.sample(master)) for weight, master in self.weights.values())

    def gradients(self):
        return {
            name: self.sample(master.grad) for name, (_, master) in self.weights.items()
            if master.grad is not None
        }


def observed_update(before, after):
    if not all(torch.isfinite(value).all().item() for pair in (*before, *after) for value in pair):
        return False
    return any(
        not torch.equal(old_weight, new_weight) and not torch.equal(old_master, new_master)
        for (old_weight, old_master), (new_weight, new_master) in zip(before, after, strict=True)
    )


def cpu_weights(snapshot):
    return tuple((weight.cpu(), master.cpu()) for weight, master in snapshot)


class ReplayTrainer(trainer.Trainer):
    def check_workload(self):
        config = self.megatron_config
        actual = (
            parallel_state.get_tensor_model_parallel_world_size(),
            parallel_state.get_pipeline_model_parallel_world_size(),
            parallel_state.get_data_parallel_world_size(),
        )
        _, pp, _ = actual
        stage = parallel_state.get_pipeline_model_parallel_rank()
        local_batch = config.global_batch_size
        if (
            actual != self.run.degrees
            or config.num_layers != LAYERS
            or config.sequence_length != self.run.length
            or local_batch != BATCH
            or config.num_micro_batches != MICROBATCHES
            or config.num_iters != self.run.steps
        ):
            raise RuntimeError(f"Unexpected training workload for seq={self.run.length}, layout={actual}.")
        if pp > 1 and (
            not config.batch_p2p_comm or config.overlap_p2p_comm or config.use_ring_exchange_p2p
        ):
            raise RuntimeError("Expected synchronous batched pipeline communication.")
        model = get_attr_wrapped_model(self.model[0], "decoder", return_model_obj=True)
        actual_layers = [layer.layer_number for layer in model.decoder.layers]
        first_layer = 1 + sum(self.run.stage_layers[:stage])
        expected_layers = list(range(first_layer, first_layer + self.run.stage_layers[stage]))
        if actual_layers != expected_layers:
            raise RuntimeError(f"Stage {stage} built layers {actual_layers}, expected {expected_layers}.")
        return model


class VerifiedTrainer(ReplayTrainer):
    def train(self, metric_types=()):
        model = self.check_workload()
        tp, pp, _ = self.run.degrees
        group = parallel_state.get_tensor_model_parallel_group()
        rank = dist.get_rank()
        samples = WeightSamples(self.optimizer, model)
        before = samples.snapshot()
        original_step = self.optimizer.optimizer.step
        unscale = self.optimizer._unscale_main_grads_and_check_for_nan
        reductions = {"total": 0, "strided": 0}
        p2p_counts = {"send": 0, "recv": 0}
        gradients = {}
        scheduled_before = self.opt_param_scheduler.num_steps
        loss = LossCheck(model, group) if model.post_process else None

        reduce = layers._reduce
        strided_ok = None
        if tp > 1:
            data = torch.arange(12, dtype=torch.float32, device=before[0][0].device)
            probe = (data + group.rank()).view(3, 4).t()
            strided_ok = torch.equal(reduce(probe), (2 * data + 1).view(3, 4).t())

        def observe_reduce(tensor):
            reductions["total"] += 1
            reductions["strided"] += int(not tensor.is_contiguous())
            return reduce(tensor)

        native_p2p = dist.batch_isend_irecv

        def observe_pipeline(ops):
            p2p_counts["send"] += sum(op.op is dist.isend for op in ops)
            p2p_counts["recv"] += sum(op.op is dist.irecv for op in ops)
            return native_p2p(ops)

        def trace_unscale():
            gradients["inverse"] = self.optimizer.grad_scaler.inv_scale.item()
            gradients["before_unscale"] = samples.gradients()
            result = unscale()
            gradients["after_unscale"] = samples.gradients()
            return result

        steps = []

        def trace_optimizer(*args, **kwargs):
            steps.append(samples.gradients())
            return original_step(*args, **kwargs)

        working_biases = set()

        def count_bias(index):
            def hook(_grad):
                working_biases.add(index)
            return hook

        checked_biases = 0
        with ExitStack() as stack:
            if loss is not None:
                stack.enter_context(patch.object(model, "compute_language_model_loss", loss))
            if tp > 1:
                stack.enter_context(patch.object(layers, "_reduce", observe_reduce))
            if pp > 1:
                stack.enter_context(patch.object(dist, "batch_isend_irecv", observe_pipeline))
            for index, layer in enumerate(model.decoder.layers):
                bias = layer.mlp.linear_fc1.bias
                if bias is not None and bias.requires_grad:
                    handle = bias.register_hook(count_bias(index))
                    stack.callback(handle.remove)
                    checked_biases += 1
            stack.enter_context(patch.object(self.optimizer, "_unscale_main_grads_and_check_for_nan", trace_unscale))
            stack.enter_context(patch.object(self.optimizer.optimizer, "step", trace_optimizer))
            result = super().train(metric_types=["train_step"])

        after = samples.snapshot()
        before_grads = gradients.get("before_unscale", {})
        after_grads = gradients.get("after_unscale", {})
        grad_unchanged = (
            gradients.get("inverse") == 1.0
            and before_grads.keys() == after_grads.keys() == samples.weights.keys()
            and all(torch.equal(before_grads[name].cpu(), after_grads[name].cpu()) for name in before_grads)
        )
        final_grads = steps[0] if len(steps) == 1 else {}
        valid_grads = (
            final_grads.keys() == samples.weights.keys()
            and all(torch.isfinite(value).all().item() for value in final_grads.values())
            and any(value.abs().max().item() > 0 for value in final_grads.values())
        )
        copied = all(torch.equal(weight, master.to(weight.dtype)) for weight, master in after)
        weight_updated = observed_update(cpu_weights(before), cpu_weights(after))
        updated_samples = self.opt_param_scheduler.num_steps - scheduled_before
        verified = (
            (loss is None or (loss.forward_checked and loss.backward_checked))
            and (tp == 1 or (reductions["total"] > 0 and (strided_ok or reductions["strided"] == 0)))
            and (pp == 1 or all(p2p_counts.values()))
            and grad_unchanged
            and valid_grads
            and copied
            and self.optimizer.found_inf.item() == 0.0
            and updated_samples == BATCH
            and weight_updated
        )
        print(
            "TEST5_MR_4L_VERIFY",
            json.dumps({
                "seq": self.run.length, "case": self.run.label, "rank": rank,
                "tp_pp_dp": self.run.degrees, "passed": verified,
                "loss_max_abs": getattr(loss, "loss_error", None),
                "logits_grad_max_abs": getattr(loss, "gradient_error", None),
                "strided_probe_ok": strided_ok, "tp_reductions": reductions,
                "p2p": p2p_counts,
                "unit_unscale_ok": grad_unchanged, "finite_nonzero_grad": valid_grads,
                "copied_model_weights": copied, "observed_weight_update": weight_updated,
                "updated_samples": updated_samples,
                "fc1_bias_grads": f"{len(working_biases)}/{checked_biases}",
            }),
            flush=True,
        )
        dist.barrier()
        if not verified:
            raise RuntimeError(f"Training verification failed for seq={self.run.length}, {self.run.label}, rank={rank}.")
        return result


class TimedTrainer(ReplayTrainer):
    def train_step(self):
        dist.barrier()
        torch.cuda.synchronize()
        start = perf_counter()
        result = super().train_step()
        torch.cuda.synchronize()
        self.timings.append((perf_counter() - start, self.timer._records["train_step"][-1].elapsed))
        # Snapshot weights outside the measured step and examine them after the run.
        self.snapshots.append(self.samples.snapshot())
        return result

    def train(self, metric_types=()):
        model = self.check_workload()
        self.samples = WeightSamples(self.optimizer, model)
        self.timings = []
        self.snapshots = [self.samples.snapshot()]
        scheduled_before = self.opt_param_scheduler.num_steps
        losses, _ = super().train(metric_types=["train_step"])
        snapshots = [cpu_weights(snapshot) for snapshot in self.snapshots]
        updates = sum(
            observed_update(before, after)
            for before, after in zip(snapshots[:-1], snapshots[1:], strict=True)
        )
        losses_ok = (
            all(
                step is not None
                and len(step) == self.megatron_config.num_micro_batches
                and all(math.isfinite(entry["lm loss"].item()) for entry in step)
                for step in losses
            ) if model.post_process else all(step is None for step in losses)
        )
        valid = (
            updates == len(losses) == len(self.timings) == self.run.steps
            and self.opt_param_scheduler.num_steps - scheduled_before == BATCH * self.run.steps
            and losses_ok
            and all(0 < fb <= full < math.inf for full, fb in self.timings)
        )
        rank = dist.get_rank()
        print(
            f"TEST5_MR_TIMED_UPDATES seq={self.run.length} case={self.run.label} "
            f"round={self.run.pair} rank={rank} verified={updates}/{self.run.steps} accepted={valid}",
            flush=True,
        )
        device = next(model.parameters()).device
        accepted = torch.tensor(float(valid), device=device)
        dist.all_reduce(accepted, op=dist.ReduceOp.SUM)
        if accepted.item() != dist.get_world_size():
            raise RuntimeError(f"Rejected timing for seq={self.run.length}, {self.run.label}.")

        times = torch.tensor(self.timings[WARMUP:], dtype=torch.float32, device=device)
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        full, fb = times.cpu().T.tolist()
        if rank == 0:
            result = {
                "sequence_length": self.run.length,
                "case": self.run.label,
                "round": self.run.pair,
                "tp_pp_dp": self.run.degrees,
                "stage_layers": self.run.stage_layers,
                "tokens_per_step": BATCH * self.run.length,
                "optimizer_updates": updates,
                "step_seconds": full,
                "forward_backward_seconds": fb,
            }
            self.run.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return losses, mean(full)


def partition_layers(layers_by_stage):
    original = transformer_block.TransformerBlock._build_layers

    def stage():
        return parallel_state.get_pipeline_model_parallel_rank()

    def layer_count(config):
        if config.pipeline_model_parallel_size != len(layers_by_stage) or config.num_layers != sum(layers_by_stage):
            raise RuntimeError(f"Invalid pipeline stage partition {layers_by_stage}.")
        return layers_by_stage[stage()]

    def layer_indices(count):
        first_index = sum(layers_by_stage[:stage()])
        return range(first_index, first_index + count)

    source = dedent(inspect.getsource(original))
    for old, new in (
        ("self.config.num_layers // global_configs.pipeline_model_parallel_size", "layer_count(self.config)"),
        ("range(num_layers)", "layer_indices(num_layers)"),
    ):
        if source.count(old) != 1:
            raise RuntimeError("Unable to install the selected layer split in the saved Crius decoder.")
        source = source.replace(old, new, 1)
    namespace = dict(vars(transformer_block), layer_count=layer_count, layer_indices=layer_indices)
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    return namespace["_build_layers"]


@record
def main():
    mode, label, pair, plan_file, output_file, *workload = sys.argv[1:]
    plan = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    degrees = tuple(plan["tp_pp_dp"])
    stage_layers = plan["stage_layers"]
    tp, pp, dp = degrees
    if label != plan["case"] or mode not in ("verify", "timed"):
        raise RuntimeError(f"Unexpected replay {mode} {label}.")
    if (
        degrees not in ((2, 1, 1), (1, 2, 1))
        or len(stage_layers) != pp
        or min(stage_layers) < 1
        or sum(stage_layers) != LAYERS
        or plan["sequence_length"] != 256
    ):
        raise RuntimeError(f"Invalid replay plan {plan}.")
    steps = 1 if mode == "verify" else WARMUP + TIMED
    run = SimpleNamespace(
        length=plan["sequence_length"], degrees=degrees, stage_layers=stage_layers,
        label=label, pair=int(pair), output=Path(output_file), steps=steps,
    )
    benchmark = VerifiedTrainer if mode == "verify" else TimedTrainer
    benchmark.run = run

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.current_device() != local_rank:
        raise RuntimeError(f"Unexpected current device on local rank {local_rank}.")
    torch.manual_seed(1234)
    validate_tp = arena_mr_vocab_check.validate_training_model

    def validate_model(model):
        if type(model).compute_language_model_loss is not LanguageModule.compute_language_model_loss:
            raise RuntimeError("The saved source is missing the corrected language-model loss.")
        if tp > 1:
            validate_tp(model)
        elif model.post_process and model.output_layer.weight.shape[0] != model.vocab_size:
            raise RuntimeError("Expected the full output vocabulary at TP=1.")

    sys.argv = [
        "scripts/bench_megatron.py", "--tp-degree", str(tp), "--pp-degree", str(pp),
        "--dp-degree", str(dp), *workload, "--num-iters", str(steps),
    ]
    try:
        with ExitStack() as stack:
            install_unit_loss_scale(stack)
            if pp > 1:
                stack.enter_context(
                    patch.object(transformer_block.TransformerBlock, "_build_layers", partition_layers(stage_layers))
                )
            stack.enter_context(patch.object(arena_mr_vocab_check, "validate_training_model", validate_model))
            stack.enter_context(patch.object(trainer, "Trainer", benchmark))
            stack.enter_context(patch.object(trainer, "time", SimpleNamespace(sleep=lambda _: None)))
            runpy.run_path("scripts/bench_megatron.py", run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY_TRAIN

cat > /tmp/arena_mr_vocab_compare.py <<'PY_COMPARE'
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def launch(output, mode, label, pair, plan_file, result_file, workload):
    log_name = f"verify-{label}.log" if mode == "verify" else f"{pair}-{label}.log"
    with (output / log_name).open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                "timeout", "-k", "15s", "900s", sys.executable, "-u", "-m", "torch.distributed.run",
                "--standalone", "--nnodes=1", "--nproc-per-node=2",
                "/tmp/arena_mr_vocab_train.py", mode, label, str(pair), str(plan_file), str(result_file),
                *workload,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"Training exited {completed.returncode}; see {log_name}.")


def check_result(run, label, pair, plan):
    actual = (
        run["case"], run["round"], run["sequence_length"], run["tokens_per_step"],
        run["tp_pp_dp"], run["stage_layers"], run["optimizer_updates"],
    )
    if actual != (label, pair, 256, 2048, plan["tp_pp_dp"], plan["stage_layers"], 7):
        raise RuntimeError(f"Wrong training workload or incomplete weight updates for {label}: {run}.")
    full, forward_backward = run["step_seconds"], run["forward_backward_seconds"]
    if len(full) != 5 or len(forward_backward) != 5 or not all(
        0 < fb <= step < math.inf for step, fb in zip(full, forward_backward, strict=True)
    ):
        raise RuntimeError(f"Invalid timing samples for {label}: {run}.")


def summarize(plan, runs):
    step_seconds = mean(value for run in runs for value in run["step_seconds"])
    return {
        "tp_pp_dp": plan["tp_pp_dp"],
        "stage_layers": plan["stage_layers"],
        "step_ms": 1000 * step_seconds,
        "forward_backward_ms": 1000 * mean(value for run in runs for value in run["forward_backward_seconds"]),
        "tokens_per_second": 2048 / step_seconds,
        "round_tokens_per_second": [2048 / mean(run["step_seconds"]) for run in runs],
        "verified_optimizer_updates_per_rank_per_run": 7,
    }


def main():
    output = Path(sys.argv[1])
    workload = sys.argv[2:]
    scores = read_json(output / "candidates.json")
    selection = read_json(output / "selection.json")
    plans = {
        score["case"]: score
        for score in scores if score["estimated_forward_backward_ms"] is not None
    }
    labels = [label for label in ("tp2", "pp2_uniform", "pp2_auto") if label in plans]
    failures = {}
    measurements = {label: [] for label in labels}

    def record_failure(label, phase, error):
        failures[label] = {"case": label, "phase": phase, "error": str(error)}
        print("TEST5_MR_4L_FAILURE", json.dumps(failures[label]), flush=True)

    for label in labels:
        plan_file = output / f"{label}-plan.json"
        plan = {**plans[label], "sequence_length": 256}
        plan_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"TEST5_MR_4L_START case={label} phase=verify", flush=True)
        try:
            launch(output, "verify", label, 0, plan_file, output / "verify-unused.json", workload)
        except RuntimeError as error:
            record_failure(label, "verify", error)

    for pair, order in enumerate((labels, list(reversed(labels))), 1):
        for label in order:
            if label in failures:
                continue
            print(f"TEST5_MR_4L_START case={label} phase=timed round={pair}", flush=True)
            result_file = output / f"{pair}-{label}.json"
            try:
                launch(output, "timed", label, pair, output / f"{label}-plan.json", result_file, workload)
                result = read_json(result_file)
                check_result(result, label, pair, plans[label])
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                record_failure(label, f"timed round {pair}", error)
                continue
            measurements[label].append(result)

    results = {
        label: summarize(plans[label], runs)
        for label, runs in measurements.items() if len(runs) == 2
    }
    selected = selection["after_vocab"]["case"]
    change = None
    if selected != "tp2" and selected in results and "tp2" in results:
        change = 100 * (results[selected]["tokens_per_second"] / results["tp2"]["tokens_per_second"] - 1)
    manual = "pp2_uniform"
    manual_change = None
    if selected in results and manual in results:
        manual_change = 100 * (results[selected]["tokens_per_second"] / results[manual]["tokens_per_second"] - 1)
    summary = {
        "num_layers": 4,
        "sequence_length": 256,
        "global_batch_size": 8,
        "transport": "native_nccl",
        "profile_source": (output / "profile-source.txt").read_text(encoding="utf-8").strip(),
        "vocab_communication_profile": read_json(output / "vocab-communication.json"),
        "rounds": 2,
        "warmup_steps_per_run": 2,
        "timed_steps_per_run": 5,
        "candidate_scores": scores,
        "arena_selected_before_vocab": selection["before_vocab"],
        "arena_selected": selection["after_vocab"],
        "fastest_measured_layout": max(results, key=lambda label: results[label]["tokens_per_second"]) if results else None,
        "measured_layouts": results,
        "manual_layout": manual,
        "arena_vs_manual_throughput_pct": manual_change,
        "arena_vs_tp2_throughput_pct": change,
        "runtime_failures": list(failures.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TEST5_MR_4L_VOCAB_COMPARISON", json.dumps(summary), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
PY_COMPARE

export ARENA_MR_VOCAB_OUTPUT=/workspace/test5-results
results="$ARENA_MR_VOCAB_OUTPUT"
profile="$results/profile-v6.pkl"
if [[ -s "$profile" ]] &&     python3 -u /tmp/arena_mr_vocab_profile.py check "$profile" 256 > "$results/profile-check.log" 2>&1; then
  printf 'TEST5_MR_4L_PROFILE_SOURCE %s\n' "$(cat "$results/profile-source.txt")"
else
  profile="$results/profile-corex.pkl"
  printf '%s\n' fresh_mr_v100_seq256 > "$results/profile-source.txt"
  printf 'TEST5_MR_4L_PROFILE_SOURCE fresh_mr_v100_seq256\n'
  for operation in all_reduce p2p; do
    timeout -k 15s 600s python3 -u -m torch.distributed.run       --standalone --nnodes=1 --nproc-per-node=2 /tmp/arena_mr_vocab_profile.py       "$operation" "$profile" 256 2>&1 | tee "$results/$operation.log"
  done
  timeout -k 15s 1200s python3 -u /tmp/arena_mr_vocab_profile.py compute "$profile" 256     2>&1 | tee "$results/profile.log"
fi

workload=(
  --batch-size 8 --seq-len 256 --num-layers 4 --hidden-size 4096
  --num-attention-heads 32 --vocab-size 50000 --num-micro-batches 4
)
timeout -k 15s 600s python3 -u -m torch.distributed.run   --standalone --nnodes=1 --nproc-per-node=2 /tmp/arena_mr_vocab_measure.py   "$results/vocab-communication.json" 2>&1 | tee "$results/vocab-communication.log"

timeout -k 15s 900s python3 -u -m torch.distributed.run   --standalone --nnodes=1 --nproc-per-node=1 /tmp/arena_mr_vocab_tune.py   --num-mlus 2 "${workload[@]}" --profiled-data-path "$profile" 2>&1 | tee "$results/tuner.log"

python3 -u /tmp/arena_mr_vocab_compare.py "$results" --num-nodes 1 "${workload[@]}"
CONTAINER

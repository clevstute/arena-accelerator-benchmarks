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
import torch_npu  # Registers the NPU backend.
from torch_npu.contrib import transfer_to_npu  # Adapts legacy Megatron CUDA calls.
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
    torch.npu.set_device(0)
    global_configs.per_device_memory_capacity_in_gb = min(
        torch.npu.get_device_properties(i).total_memory for i in range(2)
    ) / 1024**3

    output = Path(os.environ["ARENA_ASCEND_TEST5_OUTPUT"])
    sample = json.loads((output / "vocab-communication.json").read_text(encoding="utf-8"))
    vocab_ms = finite_latency(sample["per_microbatch_ms"])
    if (
        (sample["tp_degree"], sample["tensor_shape"], sample["dtype"]) != (2, [256, 2], "float32")
        or sample["reductions"] != ["max", "sum", "sum"]
        or sample["transport"] != "native_hccl"
        or (sample["warmup_samples"], sample["timed_samples"]) != (10, 50)
        or vocab_ms is None
    ):
        raise RuntimeError(f"Vocabulary communication sample does not match the comparison: {sample}.")

    entry = Path("scripts/bench_parallel_tuner.py").resolve()
    source = entry.read_text(encoding="utf-8")
    device_line = re.compile(
        r'^([ \t]*)global_configs\.current_device = torch\.device\(device\)[ \t]*$', re.M
    )
    source, matches = device_line.subn(r'\1device = "npu"\n\g<0>', source)
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
        print("TEST5_ASCEND_4L_VOCAB_CANDIDATE", json.dumps(proposal), flush=True)
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
        print("TEST5_ASCEND_4L_VOCAB_SELECTED", json.dumps(selection), flush=True)
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

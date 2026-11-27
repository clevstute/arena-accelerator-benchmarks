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
import torch_npu  # Registers the NPU backend.
from torch_npu.contrib import transfer_to_npu  # Adapts legacy Megatron CUDA calls.
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

import arena_ascend_vocab_check
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
            raise RuntimeError("Expected a constant loss scale of 1.0 for the Ascend replay.")
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
        if dist.get_backend() != "hccl":
            raise RuntimeError("Test 5 replay requires the HCCL backend.")
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
        if any(param.device.type != "npu" for param in model.parameters()):
            raise RuntimeError("Test 5 model parameters must reside on the NPU.")
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
            "TEST5_ASCEND_4L_VERIFY",
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
        torch.npu.synchronize()
        start = perf_counter()
        result = super().train_step()
        torch.npu.synchronize()
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
            f"TEST5_ASCEND_TIMED_UPDATES seq={self.run.length} case={self.run.label} "
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
    torch.npu.set_device(local_rank)
    if torch.npu.current_device() != local_rank:
        raise RuntimeError(f"Unexpected current device on local rank {local_rank}.")
    torch.manual_seed(1234)
    validate_tp = arena_ascend_vocab_check.validate_training_model

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
            stack.enter_context(patch.object(arena_ascend_vocab_check, "validate_training_model", validate_model))
            stack.enter_context(patch.object(trainer, "Trainer", benchmark))
            stack.enter_context(patch.object(trainer, "time", SimpleNamespace(sleep=lambda _: None)))
            runpy.run_path("scripts/bench_megatron.py", run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

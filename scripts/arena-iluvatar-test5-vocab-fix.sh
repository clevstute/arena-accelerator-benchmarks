#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/node-config.sh"

prepare_only=0
if [[ "${1:-}" == prepare ]]; then
    prepare_only=1
    shift
fi
if (( $# != 2 )); then
    printf 'Usage: bash %s {prepare SOURCE_DIR NEW_DIR | ORIGINAL_RUN_DIR PATH_TO_TEST5_V6_SCRIPT}\n' "$0" >&2
    exit 2
fi

original=$(cd -- "$1" && pwd -P)
test -d "$original/crius-torch"
test -d "$original/Megatron-LM"

if (( prepare_only )); then
    mkdir -- "$2"
    output=$(cd -- "$2" && pwd -P)
else
    runner=$(realpath -- "$2")
    test -s "$runner"
    test -s "$original/plan.env"
    mkdir -p -- "$arena_mr_trial_dir"
    output=$(mktemp -d "$arena_mr_trial_dir/arena-iluvatar-test5-vocab.XXXXXX")
    cp -- "$original/plan.env" "$output/"
fi
cp -a -- "$original/crius-torch" "$original/Megatron-LM" "$output/"
printf 'TEST5_MR_CORRECTED_DIR %s\n' "$output"
python3 - "$output" <<'PY'
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "crius-torch"
model_path = root / "crius/train_backend/megatron/gpt_model.py"
bench_path = root / "scripts/bench_megatron.py"
model_source = model_path.read_text(encoding="utf-8")
bench_source = bench_path.read_text(encoding="utf-8")
model = next(
    node for node in ast.parse(model_source).body
    if isinstance(node, ast.ClassDef) and node.name == "GPTModel"
)
if not any(isinstance(base, ast.Name) and base.id == "LanguageModule" for base in model.bases):
    raise SystemExit("Unexpected GPTModel base class.")
loss = next(
    (node for node in model.body
     if isinstance(node, ast.FunctionDef) and node.name == "compute_language_model_loss"),
    None,
)
if loss is None or "torch.nn.CrossEntropyLoss()" not in ast.get_source_segment(model_source, loss):
    raise SystemExit("Cannot identify the original GPT loss override.")
lines = model_source.splitlines(keepends=True)
del lines[loss.lineno - 1:loss.end_lineno]
updated_model = "".join(lines).rstrip() + "\n"
setup = "    model = setup_model(model)\n"
if bench_source.count(setup) != 1:
    raise SystemExit("Cannot identify the benchmark model setup.")
updated_bench = bench_source.replace(
    setup,
    "    from arena_mr_vocab_check import validate_training_model\n"
    "    validate_training_model(model)\n"
    + setup,
    1,
)
for path, updated in ((model_path, updated_model), (bench_path, updated_bench)):
    compile(updated, str(path), "exec")
for path, updated in ((model_path, updated_model), (bench_path, updated_bench)):
    path.write_text(updated, encoding="utf-8")
PY

cat > "$output/crius-torch/arena_mr_vocab_check.py" <<'PY'
import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.language_module.language_module import LanguageModule
from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear


def validate_training_model(model):
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size != 2:
        raise RuntimeError(f"Expected the paired TP=2 replay, got TP={tp_size}.")
    if type(model).compute_language_model_loss is not LanguageModule.compute_language_model_loss:
        raise RuntimeError("GPT is not using the Megatron language-model loss.")

    def mark(parameter, dimension):
        expected = (True, dimension, 1)
        if hasattr(parameter, "tensor_model_parallel"):
            found = (
                parameter.tensor_model_parallel,
                parameter.partition_dim,
                parameter.partition_stride,
            )
            if found != expected:
                raise RuntimeError(f"Wrong TP weight metadata {found}, expected {expected}.")
        else:
            tensor_parallel.set_tensor_model_parallel_attributes(parameter, *expected)

    for layer in model.modules():
        if isinstance(layer, ColumnParallelLinear):
            mark(layer.weight, 0)
            if layer.bias is not None:
                mark(layer.bias, 0)
        elif isinstance(layer, RowParallelLinear):
            mark(layer.weight, 1)

    device = model.output_layer.weight.device
    rank = parallel_state.get_tensor_model_parallel_rank()
    if model.output_layer.weight.shape[0] * tp_size != model.vocab_size:
        raise RuntimeError("Invalid output vocabulary partition.")
    logits = torch.tensor([
        [[-0.5, 0.75, 1.5, 0.125, -1.5, 0.25, 0.625, 0.375]],
        [[1.25, -0.125, -1.5, 1.125, -0.5, 0.75, -0.25, 1.625]],
        [[0.25, 0.5, -0.75, -1.0, 1.125, 1.25, -1.625, -0.875]],
        [[-0.875, 1.125, 0.625, 0.875, 1.5, -1.625, 0.125, -1.5]],
    ], dtype=torch.float32, device="cpu").requires_grad_()
    labels = torch.tensor([[1, 7, 4, 3]], device="cpu")
    shard = slice(rank * 4, (rank + 1) * 4)
    local_logits = logits.detach()[..., shard].contiguous().to(
        device=device, dtype=torch.float16
    ).requires_grad_()

    expected_loss = F.cross_entropy(logits.permute(1, 2, 0), labels, reduction="none")
    expected_loss.sum().backward()
    actual_loss = model.compute_language_model_loss(labels.to(device), local_logits)
    actual_loss.sum().backward()
    torch.testing.assert_close(
        actual_loss.detach().cpu(), expected_loss.detach(), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        local_logits.grad.cpu(), logits.grad[..., shard].to(torch.float16),
        rtol=2e-3, atol=1e-5,
    )
    print(f"TEST5_MR_VOCAB_PASS tp_rank={rank} device={device} loss_and_gradient=ok", flush=True)
PY

if (( ! prepare_only )); then
    bash "$runner" compare "$output"
fi

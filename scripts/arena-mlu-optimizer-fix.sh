#!/usr/bin/env bash
set -e -o pipefail

trial_dir="${ARENA_MLU_TRIAL_DIR:-$HOME/arena-mlu-work}"
bundle_dir="$trial_dir/arena-mlu-train-smoke"
source "$trial_dir/.venv/bin/activate"

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - "$bundle_dir" <<'PY'
import shutil
import sys
import sysconfig
from pathlib import Path

bundle = Path(sys.argv[1])
relative = Path('megatron/core/optimizer')
roots = (bundle / 'Megatron-LM' / relative, Path(sysconfig.get_path('platlib')) / relative)

edits = {
    'optimizer.py': (
        (
            "if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:",
            "if param.device.type in ('cuda', 'mlu') and param.dtype in (\n"
            "                        torch.float16, torch.bfloat16\n"
            "                    ):",
        ),
        (
            "elif param.type() == 'torch.cuda.FloatTensor':",
            "elif param.device.type in ('cuda', 'mlu') and param.dtype == torch.float32:",
        ),
        (
            "'Wrapped parameters must be one of '\n"
            "                            'torch.cuda.FloatTensor,  '\n"
            "                            'torch.cuda.HalfTensor, or '\n"
            "                            'torch.cuda.BFloat16Tensor. '\n"
            "                            'Received {}'.format(param.type())",
            "'Expected float16, bfloat16 or float32 parameters on CUDA or MLU. '\n"
            "                            'Received {}'.format(param.type())",
        ),
    ),
    'clip_grads.py': (
        (
            "assert param.grad.type() == 'torch.cuda.FloatTensor'",
            "assert param.grad.device.type in ('cuda', 'mlu') and param.grad.dtype == torch.float32",
        ),
    ),
}

pending = []
for root in roots:
    for filename, replacements in edits.items():
        path = root / filename
        original = path.read_text()
        updated = original
        for old, new in replacements:
            if updated.count(old) == 1:
                updated = updated.replace(old, new, 1)
            elif updated.count(old) != 0 or updated.count(new) != 1:
                raise SystemExit(f'Unexpected Megatron source in {path}. No changes written.')
        if updated != original:
            compile(updated, str(path), 'exec')
            pending.append((path, updated))

for path, updated in pending:
    backup = path.with_name(path.name + '.before-arena-mlu-fix')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated)
    print(f'Updated {path}')
PY

cat > "$bundle_dir/run-optimizer-smoke-on-mlu.py" <<'PY'
import runpy
import sys
from functools import wraps
from pathlib import Path

from torch.distributed.elastic.multiprocessing.errors import record


@record
def main():
    import torch
    import torch_mlu.utils.gpu_migration  # Enables the MLU backend for legacy CUDA calls.

    from crius.train_backend.megatron.training import utils

    assert torch.mlu.is_available(), 'PyTorch did not detect the MLU'
    setup_optimizer = utils.get_optimizer_and_scheduler

    @wraps(setup_optimizer)
    def setup_for_smoke_test(*args, **kwargs):
        # Keep the three-step smoke test independent of loss-scale warmup.
        kwargs['loss_scale'] = 1.0
        optimizer, scheduler = setup_optimizer(*args, **kwargs)
        optimizer_step = optimizer.step

        @wraps(optimizer_step)
        def step_and_report(*step_args, **step_kwargs):
            outcome = optimizer_step(*step_args, **step_kwargs)
            status = 'applied' if outcome[0] else 'skipped'
            print(f'Optimizer update {status}', flush=True)
            return outcome

        optimizer.step = step_and_report
        return optimizer, scheduler

    utils.get_optimizer_and_scheduler = setup_for_smoke_test
    entry = Path(__file__).resolve().parent / 'crius-torch/scripts/bench_megatron.py'
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
PY

cd "$bundle_dir/crius-torch"
log="$bundle_dir/arena-smoke-optimizer.log"
python -m torch.distributed.run --standalone --nproc-per-node=1 \
    "$bundle_dir/run-optimizer-smoke-on-mlu.py" \
    --tp-degree 1 --pp-degree 1 --dp-degree 1 \
    --batch-size 2 --seq-len 64 --num-layers 2 \
    --hidden-size 256 --num-attention-heads 8 \
    --num-micro-batches 2 --num-iters 3 2>&1 | tee "$log"

if ! grep -q '^Optimizer update applied$' "$log"; then
    printf '%s\n' 'No optimizer update was applied. See the smoke test log.' >&2
    exit 1
fi

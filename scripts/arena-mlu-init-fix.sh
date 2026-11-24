#!/usr/bin/env bash
set -e -o pipefail

trial_dir="${ARENA_MLU_TRIAL_DIR:-$HOME/arena-mlu-work}"
bundle_dir="$trial_dir/arena-mlu-train-smoke"
python_bin="$trial_dir/.venv/bin/python"

"$python_bin" - "$bundle_dir/crius-torch/scripts/bench_megatron.py" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
original = path.read_text()
old_import = '    from crius.train_backend.megatron.gpt_model import GPTModel\n'
new_import = old_import + '    from crius.train_backend.megatron.layers import ColumnParallelLinear, RowParallelLinear\n'
old_setup = '    model = setup_model(model)\n'
new_setup = '''    if megatron_config.perform_initialization:
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, ColumnParallelLinear):
                    megatron_config.init_method(layer.weight)
                elif isinstance(layer, RowParallelLinear):
                    megatron_config.output_layer_init_method(layer.weight)
                else:
                    continue
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

''' + old_setup

if original.count(new_import) == 1 and original.count(new_setup) == 1:
    print(f'Megatron parameter initialization already added to {path}')
else:
    if new_import in original or new_setup in original or original.count(old_import) != 1 or original.count(old_setup) != 1:
        raise SystemExit(f'Unexpected benchmark source in {path}. No changes written.')

    updated = original.replace(old_import, new_import, 1).replace(old_setup, new_setup, 1)
    compile(updated, str(path), 'exec')
    backup = path.with_name(path.name + '.before-arena-init-fix')
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_name(path.name + '.arena-init.tmp')
    temporary.write_text(updated)
    shutil.copymode(path, temporary)
    temporary.replace(path)
    print(f'Added Megatron parameter initialization to {path}')
PY

bash "$(dirname -- "${BASH_SOURCE[0]}")/arena-mlu-optimizer-fix.sh"

"$python_bin" - "$bundle_dir/arena-smoke-optimizer.log" <<'PY'
import math
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
iterations = re.findall(r'Losses of the training iteration \((\d+)\): (.+)', log)
if [int(index) for index, _ in iterations] != [0, 1, 2]:
    raise SystemExit('Expected three complete training iterations. See the smoke test log.')

for index, summary in iterations:
    losses = re.findall(r"'lm loss': tensor\(([^,)]+)", summary)
    if len(losses) != 2 or any(not math.isfinite(float(loss)) for loss in losses):
        raise SystemExit(f'Iteration {index} did not produce two finite microbatch losses.')

updates = log.splitlines().count('Optimizer update applied')
if updates != 3:
    raise SystemExit(f'Only {updates} of 3 optimizer updates were applied.')

print('Single-card smoke test passed: finite losses and 3 optimizer updates.')
PY

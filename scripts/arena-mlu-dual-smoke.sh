#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
case "$action" in
    pack|install|run) ;;
    *) printf 'Usage: bash %s {pack|install|run}\n' "${0##*/}" >&2; exit 2 ;;
esac

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/node-config.sh"

bundle_dir="$trial_dir/arena-mlu-train-smoke"
archive="$HOME/arena-mlu-dual-input.tar.gz"
source "$trial_dir/.venv/bin/activate"
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "$action" in
    pack)
        if [[ "$rank" != 0 ]]; then
            printf '%s\n' 'Pack the tested code on rank 0.' >&2
            exit 1
        fi
        stage_dir="$(mktemp -d "$trial_dir/dual-pack.XXXXXX")"
        trap 'rm -rf -- "$stage_dir"' EXIT
        mkdir "$stage_dir/wheels"
        CC=/usr/bin/gcc CXX=/usr/bin/g++ python -m pip wheel \
            --no-build-isolation --no-deps --wheel-dir "$stage_dir/wheels" \
            "$bundle_dir/Megatron-LM"
        tar -C "$bundle_dir" -czf "$stage_dir/input.tar.gz" \
            crius-torch run-optimizer-smoke-on-mlu.py -C "$stage_dir" wheels
        mv -f "$stage_dir/input.tar.gz" "$archive"
        printf 'Packed trial dependencies: %s\n' "$archive"
        ;;
    install)
        if [[ "$rank" != 1 ]]; then
            printf '%s\n' 'Install the transferred code on rank 1.' >&2
            exit 1
        fi
        mkdir -p "$bundle_dir"
        tar -xzf "$archive" -C "$bundle_dir"
        python -m pip install 'setuptools==75.8.0' wheel packaging psutil sentencepiece
        python -m pip install --no-deps "$bundle_dir"/wheels/megatron_core-*.whl
        python -m pip install --no-build-isolation --no-deps -e "$bundle_dir/crius-torch"
        python -c 'import torch_mlu.utils.gpu_migration; from crius.train_backend.megatron.training import trainer'
        printf '%s\n' 'Crius and Megatron Core are ready on rank 1.'
        ;;
    run)
        arena_mlu_require_master
        arena_mlu_set_gloo_interface
        cd "$bundle_dir/crius-torch"
        log="$(mktemp "$bundle_dir/arena-dual-smoke-$host.XXXXXX.log")"
        printf 'Running TP=2, PP=1, DP=1 on %s (rank %s)\nLog: %s\n' "$host" "$rank" "$log"
        timeout -k 10s 300s python -m torch.distributed.run \
            --nnodes=2 --nproc-per-node=1 --node-rank="$rank" \
            --master-addr="$ARENA_MLU_MASTER_ADDR" --master-port=29643 --max-restarts=0 \
            "$bundle_dir/run-optimizer-smoke-on-mlu.py" \
            --num-nodes 2 --tp-degree 2 --pp-degree 1 --dp-degree 1 \
            --batch-size 2 --seq-len 64 --num-layers 2 \
            --hidden-size 256 --num-attention-heads 8 \
            --num-micro-batches 2 --num-iters 3 2>&1 | tee "$log"
        python - "$log" "$rank" <<'PY'
import math
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
rank = int(sys.argv[2])
lines = log.splitlines()
if lines.count("Optimizer update applied") != 3 or "Optimizer update skipped" in lines:
    raise SystemExit(f"Rank {rank}: expected three successful parameter updates.")

if rank == 0:
    iterations = re.findall(r"Losses of the training iteration \((\d+)\): (.+)", log)
    if [int(index) for index, _ in iterations] != [0, 1, 2]:
        raise SystemExit("Rank 0: expected three training iterations with recorded losses.")
    for index, summary in iterations:
        losses = re.findall(r"'lm loss': tensor\(([^,)]+)", summary)
        if len(losses) != 2 or any(not math.isfinite(float(loss)) for loss in losses):
            raise SystemExit(f"Rank 0: invalid losses in training iteration {index}.")

print(f"Dual-node training smoke passed on rank {rank}: 3 optimizer updates.")
PY
        ;;
esac

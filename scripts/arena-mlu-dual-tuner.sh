#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
case "$action" in
    comm|tune) ;;
    *) printf 'Usage: bash %s {comm|tune}\n' "${0##*/}" >&2; exit 2 ;;
esac

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/node-config.sh"
if [[ "$action" == tune && "$rank" != 0 ]]; then
    printf '%s\n' 'Run the tuner on rank 0.' >&2
    exit 1
fi

source "$trial_dir/.venv/bin/activate"
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
arena_mlu_set_gloo_interface
cd "$trial_dir/arena-mlu-train-smoke/crius-torch"

work_dir="$trial_dir/arena-mlu-dual-tuning"
mkdir -p "$work_dir"
run_dir="$(mktemp -d "$work_dir/$action.XXXXXX")"
profile="$work_dir/mlu580-dual-profile.pkl"
printf 'Run directory: %s\n' "$run_dir"

cat > "$run_dir/collect_mlu_profile.py" <<'PY'
import math
import sys
from pathlib import Path

import torch
import torch_mlu
import torch_mlu.utils.gpu_migration
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
    torch.mlu.set_device(0)
    heads = [16, 32]
    hidden_dims = [2048, HIDDEN, 8192, 16384]
    configs = OfflineProfileConfigs(
        device="mlu:0",
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
    required = [f"all_reduce__n2__gs2__d{n}" for n in ALL_REDUCE_ELEMENTS]
    for key in required:
        latency = records.get(key, {}).get("latency_ms")
        if not isinstance(latency, (float, int)) or not 0 < latency < math.inf:
            raise SystemExit(f"Missing or invalid cross-node profile for {key}. Run comm on both nodes.")
    p2p_key = f"p2p__n2__gs2__d{ACTIVATION_ELEMENTS}"
    latency = records.get(p2p_key, {}).get("latency_ms")
    if not isinstance(latency, list) or len(latency) != 3 or any(
        not 0 < value < math.inf for value in latency
    ):
        raise SystemExit(f"Missing or invalid cross-node profile for {p2p_key}. Run comm on both nodes.")

    for opcode in ("gemm", "bmm", "baddbmm", "layernorm", "dropout", "masked_fill_", "softmax"):
        profiler.profile(opcode)


if __name__ == "__main__":
    main()
PY

if [[ "$action" == comm ]]; then
    arena_mlu_require_master
    port=29644
    for opcode in all_reduce p2p; do
        printf 'Profiling %s on %s (rank %s)\n' "$opcode" "$host" "$rank"
        timeout -k 10s 600s python -m torch.distributed.run \
            --nnodes=2 --nproc-per-node=1 --node-rank="$rank" \
            --master-addr="$ARENA_MLU_MASTER_ADDR" --master-port="$port" --max-restarts=0 \
            "$run_dir/collect_mlu_profile.py" "$opcode" "$profile" 2>&1 | tee "$run_dir/$opcode.log"
        port=29645
    done
    printf 'Dual-node communication profiling finished on %s.\n' "$host"
    exit
fi

python -u "$run_dir/collect_mlu_profile.py" compute "$profile" 2>&1 | tee "$run_dir/compute.log"
cat > "$run_dir/tune.py" <<'PY'
import sys
from pathlib import Path

import torch
import torch_mlu
import torch_mlu.utils.gpu_migration
from crius.global_envs import global_configs
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main():
    memory_bytes = torch.mlu.get_device_properties(0).total_memory
    global_configs.per_device_memory_capacity_in_gb = memory_bytes / 1024**3
    entry = Path("scripts/bench_parallel_tuner.py").resolve()
    source = entry.read_text()
    node_setting = "    global_configs.num_nodes = 1\n"
    if source.count(node_setting) != 1:
        raise SystemExit(f"Unexpected node setting in {entry}.")
    source = source.replace(node_setting, "    global_configs.num_nodes = 2\n")
    print("Tuner topology: 2 nodes, 1 MLU580 per node", flush=True)
    sys.argv[0] = str(entry)
    exec(compile(source, str(entry), "exec"), {"__name__": "__main__", "__file__": str(entry)})


if __name__ == "__main__":
    main()
PY
python -m torch.distributed.run --standalone --nproc-per-node=1 \
    "$run_dir/tune.py" --num-mlus 1 --batch-size 8 --seq-len 256 \
    --num-layers 4 --hidden-size 4096 --num-attention-heads 32 --vocab-size 50000 \
    --num-micro-batches 4 --profiled-data-path "$profile" --env-output "$run_dir/plan.env" \
    2>&1 | tee "$run_dir/tuner.log"
printf 'Tuner plan: %s\n' "$run_dir/plan.env"

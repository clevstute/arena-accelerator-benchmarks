#!/usr/bin/env bash
set -o pipefail

phase="${1:-verify}"
if [[ "$phase" == -h || "$phase" == --help ]]; then
    printf 'Usage: bash %s [verify|vocab|run]\n' "${0##*/}"
    exit 0
fi
if (( $# > 1 )) || [[ "$phase" != verify && "$phase" != vocab && "$phase" != run ]]; then
    printf 'Usage: bash %s [verify|vocab|run]\n' "${0##*/}" >&2
    exit 2
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit
set -eu
# Each benchmark worker imports torch_npu explicitly.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
py=/opt/arena-venv/bin/python
if [[ ! -x "$py" ]]; then
    printf 'Missing the tested Ascend Python environment at %s.\n' "$py" >&2
    exit 1
fi
export PYTHONPATH="$bundle/source/crius-torch:$bundle/source/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
scripts="$bundle/scripts"

mkdir -p -- "$bundle/results"
output=$(mktemp -d "$bundle/results/$phase.XXXXXX")
export ARENA_ASCEND_TEST5_OUTPUT="$output"
printf 'TEST5_ASCEND_DIR %s\n' "$output"

"$py" - <<'PY'
import torch
import torch_npu

count = torch.npu.device_count()
if count != 2:
    raise RuntimeError(f"Expected two visible NPUs in the Arena container, found {count}.")
print(f"TEST5_ASCEND_ENV torch={torch.__version__} torch_npu={torch_npu.__version__} npus={count}")
PY

launcher=("$py" -u -m torch.distributed.run --standalone --nnodes=1)
workload=(
    --batch-size 8 --seq-len 256 --num-layers 4 --hidden-size 4096
    --num-attention-heads 32 --vocab-size 50000 --num-micro-batches 4
)

measure_vocab() {
    timeout -k 15s 300s "${launcher[@]}" --nproc-per-node=2 \
        "$scripts/vocab_communication.py" "$output/vocab-communication.json" \
        2>&1 | tee "$output/vocab-communication.log"
}

if [[ "$phase" == vocab ]]; then
    measure_vocab
    exit 0
fi

"$py" -m pip install --disable-pip-version-check --only-binary=:all: \
    sentencepiece==0.2.1 pybind11==2.13.6 packaging psutil 2>&1 | tee "$output/dependencies.log"

build_packages=()
command -v g++ >/dev/null || build_packages+=(g++)
command -v make >/dev/null || build_packages+=(make)
if (( ${#build_packages[@]} )); then
    apt-get update -qq
    apt-get install -y --no-install-recommends "${build_packages[@]}"
fi

include=$("$py" -c 'import sysconfig; print(sysconfig.get_path("include"))')
if [[ ! -r "$include/Python.h" ]]; then
    printf 'Cannot compile the Megatron dataset helper. Missing %s/Python.h.\n' "$include" >&2
    exit 1
fi
suffix=$("$py" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
includes=$("$py" -m pybind11 --includes)
make -C "$bundle/source/Megatron-LM/megatron/core/datasets" \
    "LIBEXT=$suffix" "CPPFLAGS=$includes" 2>&1 | tee "$output/dataset-build.log"

cd -- "$bundle/source/crius-torch"
if [[ "$phase" == verify ]]; then
    labels=(tp2 pp2_uniform)
    degrees=('[2,1,1]' '[1,2,1]')
    layers=('[4]' '[2,2]')
    for index in "${!labels[@]}"; do
        label="${labels[index]}"
        plan="$output/$label-plan.json"
        printf '{"case":"%s","tp_pp_dp":%s,"stage_layers":%s,"sequence_length":256}\n' \
            "$label" "${degrees[index]}" "${layers[index]}" > "$plan"
        timeout -k 15s 900s "${launcher[@]}" --nproc-per-node=2 "$scripts/replay.py" \
            verify "$label" 0 "$plan" "$output/verify-unused.json" "${workload[@]}" \
            2>&1 | tee "$output/verify-$label.log"
    done
    printf 'TEST5_ASCEND_VERIFY_OK %s\n' "$output"
    exit 0
fi

profile="$output/profile-ascend.pkl"
printf '%s\n' fresh_ascend_910b_seq256 > "$output/profile-source.txt"
for operation in all_reduce p2p; do
    timeout -k 15s 600s "${launcher[@]}" --nproc-per-node=2 \
        "$scripts/profile_workload.py" "$operation" "$profile" 256 2>&1 | tee "$output/$operation.log"
done
timeout -k 15s 1200s "$py" -u "$scripts/profile_workload.py" compute "$profile" 256 \
    2>&1 | tee "$output/profile.log"
measure_vocab

timeout -k 15s 900s "${launcher[@]}" --nproc-per-node=1 "$scripts/tune.py" \
    --num-mlus 2 "${workload[@]}" --profiled-data-path "$profile" 2>&1 | tee "$output/tuner.log"
"$py" -u "$scripts/compare.py" "$output" --num-nodes 1 "${workload[@]}"

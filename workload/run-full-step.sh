#!/usr/bin/env bash

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    printf 'Usage: bash %s [previous-run-directory]\n' "${0##*/}"
    exit 0
fi
if (( $# > 1 )); then
    printf 'Usage: bash %s [previous-run-directory]\n' "${0##*/}" >&2
    exit 2
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit
set -eu
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$bundle/source/crius-torch:$bundle/source/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
exec /opt/arena-venv/bin/python -u "$bundle/scripts/full_step.py" "$@"

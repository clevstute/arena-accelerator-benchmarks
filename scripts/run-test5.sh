#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    printf 'Usage: bash %s (configure source archive and work directory in config/node.env.example)\n' "$0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/node-config.sh"

mkdir -p -- "$arena_mr_trial_dir"
work_dir=$(mktemp -d "$arena_mr_trial_dir/arena-iluvatar-repro.XXXXXX")
work_dir=$(cd -- "$work_dir" && pwd -P)
mkdir -- "$work_dir/source"
printf 'TEST5_MR_WORK_DIR %s\n' "$work_dir"

bash "$script_dir/arena-iluvatar-test5-v6.sh" prepare "$work_dir/source"
bash "$script_dir/arena-iluvatar-test5-vocab-fix.sh" prepare "$work_dir/source" "$work_dir/vocab"
exec bash "$script_dir/arena-iluvatar-test5-4layer-vocab-cost.sh" "$work_dir/vocab"

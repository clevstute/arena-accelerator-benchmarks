#!/usr/bin/env bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit
set -eu
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
venv=/opt/arena-venv
if [[ ! -x "$venv/bin/python" ]]; then
    python3.11 -m venv --system-site-packages "$venv"
fi
"$venv/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: \
    torch==2.6.0 torch-npu==2.6.0 numpy==1.24.0 'protobuf==3.20.*' pyyaml setuptools
"$venv/bin/python" "$script_dir/fix_torch_npu_import.py"

#!/usr/bin/env bash
set -euo pipefail

image=${ARENA_ASCEND_IMAGE:-swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-910b-ubuntu22.04-py3.11}
container=${ARENA_ASCEND_CONTAINER:-arena-ascend910b3-test5}
driver=${ARENA_ASCEND_DRIVER_DIR:-/usr/local/Ascend/driver}
npu_a=${ARENA_ASCEND_NPU_A:-0}
npu_b=${ARENA_ASCEND_NPU_B:-1}
if [[ ! "$npu_a" =~ ^[0-9]+$ || ! "$npu_b" =~ ^[0-9]+$ || "$npu_a" == "$npu_b" ]]; then
    printf 'Set two distinct Ascend NPU indices.\n' >&2
    exit 2
fi

mounts=(--mount "type=bind,src=$driver,dst=/usr/local/Ascend/driver,readonly")
for path in /usr/local/dcmi /etc/ascend_install.info; do
    if [[ -e "$path" ]]; then
        mounts+=(--mount "type=bind,src=$path,dst=$path,readonly")
    fi
done

docker run --detach --pull=never --platform linux/arm64 --name "$container" \
    --device "/dev/davinci$npu_a" --device "/dev/davinci$npu_b" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    "${mounts[@]}" \
    -e DRIVER_PATH=/usr/local/Ascend/driver -e "ASCEND_RT_VISIBLE_DEVICES=$npu_a,$npu_b" \
    --shm-size=1g --entrypoint /bin/bash "$image" -c 'exec sleep infinity'

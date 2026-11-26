# Shared host configuration for the single-node MR-V100 test.
arena_mr_trial_dir="${ARENA_MR_TRIAL_DIR:-$HOME}"
arena_mr_source_archive="${ARENA_MR_SOURCE_ARCHIVE:-$HOME/arena-iluvatar-src.tar.gz}"
arena_mr_driver_dir="${ARENA_MR_DRIVER_DIR:-/usr/local/corex/lib}"
arena_mr_image="${ARENA_MR_IMAGE:-registry.iluvatar.com.cn:10443/release/pytorch:2.10.0-5.0.0_py3.12-corex.5.0.0-ubuntu24.04}"

arena_mr_docker_options() {
    local lib source
    docker_options=(
        --rm -i --pull=never --shm-size=1g
        --device=/dev/iluvatar0 --device=/dev/iluvatar1 --device=/dev/itrctl
        -e CUDA_VISIBLE_DEVICES=0,1
    )
    for lib in "$arena_mr_driver_dir"/libcuda.so* "$arena_mr_driver_dir"/libixthunk.so* "$arena_mr_driver_dir"/libixml.so*; do
        [[ -f "$lib" ]] || continue
        source=$(readlink -f -- "$lib")
        [[ "$source" != */stubs/* ]] || continue
        docker_options+=(--mount "type=bind,source=$source,target=/opt/arena-corex-driver/${lib##*/},readonly")
    done
}

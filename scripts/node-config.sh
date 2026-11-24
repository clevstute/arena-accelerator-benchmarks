# Shared node configuration for the two-node MLU580 benchmark.
host="$(hostname -s)"
rank="${ARENA_MLU_NODE_RANK:-}"
trial_dir="${ARENA_MLU_TRIAL_DIR:-$HOME/arena-mlu-work}"

case "$rank" in
    0|1) ;;
    *) printf '%s\n' 'Set ARENA_MLU_NODE_RANK=0 on the first node and =1 on the second.' >&2; exit 2 ;;
esac

arena_mlu_require_master() {
    : "${ARENA_MLU_MASTER_ADDR:?Set ARENA_MLU_MASTER_ADDR to the address of rank 0 on both nodes.}"
}

arena_mlu_set_gloo_interface() {
    if [[ -n "${ARENA_MLU_NET_IFACE:-}" ]]; then
        export GLOO_SOCKET_IFNAME="$ARENA_MLU_NET_IFACE"
    fi
}

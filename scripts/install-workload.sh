#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
container=${ARENA_ASCEND_CONTAINER:-arena-ascend910b3-test5}
target=${ARENA_ASCEND_WORKLOAD_DIR:-/root/arena-ascend-test5}
archive=${ARENA_ASCEND_SOURCE_ARCHIVE:?Set ARENA_ASCEND_SOURCE_ARCHIVE to the tested arena-ascend-test5.tar.gz.}

docker exec "$container" mkdir -p -- "$target"
docker exec -i -w "$target" "$container" tar -xzf - --strip-components=1 \
    arena-ascend-test5/source/ < "$archive"
docker cp "$root/workload/." "$container:$target/"
docker exec -w "$target" "$container" bash scripts/setup-venv.sh

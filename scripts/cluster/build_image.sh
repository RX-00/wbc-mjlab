#!/usr/bin/env bash
# Build locally, then transfer the SIF and this source snapshot to the cluster.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
BUILD_ROOT=${BUILD_ROOT:-"$REPO/logs/cluster_build_$(date +%Y-%m-%d_%H-%M-%S)"}
mkdir -p "$BUILD_ROOT/context" "$BUILD_ROOT/tmp" "$BUILD_ROOT/cache"
CONTEXT="$BUILD_ROOT/context"
# Git files, plus the local scripts and converted sample motions. No old runs.
git ls-files -z | rsync -a --from0 --files-from=- ./ "$CONTEXT/"
rsync -a --exclude=__pycache__/ scripts/ "$CONTEXT/scripts/"
mkdir -p "$CONTEXT/data/g1/samples/npz"
rsync -a data/g1/samples/npz/ "$CONTEXT/data/g1/samples/npz/"
uv export --quiet --frozen --extra cu128 --no-dev --no-emit-project \
  --format pylock.toml --output-file "$CONTEXT/pylock.toml"
git rev-parse HEAD > "$CONTEXT/SOURCE_REVISION"
git diff --binary HEAD > "$CONTEXT/SOURCE_CHANGES.patch"
uv --version > "$CONTEXT/UV_VERSION"
(cd "$CONTEXT" && find src scripts data/g1/samples/npz pyproject.toml uv.lock -type f -print0 | \
  sort -z | xargs -0 sha256sum) > "$CONTEXT/SNAPSHOT.sha256"
export APPTAINER_TMPDIR="$BUILD_ROOT/tmp"
export APPTAINER_CACHEDIR="$BUILD_ROOT/cache"
apptainer build --fakeroot \
  --build-arg "context=$CONTEXT" --build-arg "uv_binary=$(command -v uv)" \
  "$BUILD_ROOT/wbc-mjlab-cu128.sif" scripts/cluster/wbc.def
(cd "$BUILD_ROOT" && sha256sum wbc-mjlab-cu128.sif > IMAGE.sha256)
printf 'Build and snapshot: %s\n' "$BUILD_ROOT"

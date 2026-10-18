#!/bin/sh
set -eu

SOURCE_IMAGE=${SOURCE_IMAGE:-local/vllm-openai:sm61-v028}
DEST_ROOT=${DEST_ROOT:-artifacts/torch/site-packages}
DEST_SUBDIR=${DEST_SUBDIR:-extract-$(date -u +%Y%m%dT%H%M%SZ)}
DEST_DIR=$DEST_ROOT/$DEST_SUBDIR
MANIFEST=artifacts/torch/manifest.txt

mkdir -p "$DEST_DIR"

container_id=$(docker create "$SOURCE_IMAGE")
cleanup() {
  docker rm "$container_id" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

torch_dist_info=$(docker run --rm --entrypoint sh "$SOURCE_IMAGE" -c 'ls -d /usr/local/lib/python3.12/dist-packages/torch-*.dist-info 2>/dev/null | xargs -n1 basename' | head -1)
[ -n "$torch_dist_info" ] || { echo "torch dist-info not found in $SOURCE_IMAGE"; exit 1; }
echo "Extracting $torch_dist_info from $SOURCE_IMAGE"

docker cp "$container_id":/usr/local/lib/python3.12/dist-packages/torch "$DEST_DIR"/
docker cp "$container_id":/usr/local/lib/python3.12/dist-packages/torchgen "$DEST_DIR"/
docker cp "$container_id":/usr/local/lib/python3.12/dist-packages/functorch "$DEST_DIR"/
docker cp "$container_id":/usr/local/lib/python3.12/dist-packages/"$torch_dist_info" "$DEST_DIR"/

ln -sfn "$DEST_SUBDIR" "$DEST_ROOT/current"

printf '%s\n' \
  "source_image=$SOURCE_IMAGE" \
  "package_root=$DEST_DIR" \
  'torch_package=torch' \
  'torchgen_package=torchgen' \
  'functorch_package=functorch' \
  "torch_dist_info=$torch_dist_info" \
  > "$MANIFEST"

du -sh "$DEST_DIR"/*
printf 'Wrote torch artifact manifest to %s\n' "$MANIFEST"

#!/usr/bin/env bash
set -euo pipefail

CUSPARSELT_VERSION="${CUSPARSELT_VERSION:-0.6.2.3}"
ARCHIVE_URL="${ARCHIVE_URL:-https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive.tar.xz}"
INSTALL_ROOT="${INSTALL_ROOT:-${CONDA_PREFIX:-$HOME/.local/cusparselt}}"
LIB_DEST="${LIB_DEST:-${CONDA_PREFIX:-$INSTALL_ROOT}/lib}"
INCLUDE_DEST="${INCLUDE_DEST:-${INSTALL_ROOT}/include}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

echo "Downloading cuSPARSELt archive:"
echo "  $ARCHIVE_URL"

curl -fsSLo "$tmp_dir/cusparselt.tar.xz" "$ARCHIVE_URL"
tar -C "$tmp_dir" -xf "$tmp_dir/cusparselt.tar.xz"

archive_dir="$(find "$tmp_dir" -maxdepth 1 -type d -name 'libcusparse_lt-linux-aarch64-*' | head -n 1)"
if [[ -z "$archive_dir" ]]; then
  echo "Could not find the extracted cuSPARSELt archive directory." >&2
  exit 1
fi

mkdir -p "$LIB_DEST" "$INCLUDE_DEST"
cp -a "$archive_dir/lib/." "$LIB_DEST/"
cp -a "$archive_dir/include/." "$INCLUDE_DEST/"

echo
echo "Installed cuSPARSELt into:"
echo "  lib:     $LIB_DEST"
echo "  include: $INCLUDE_DEST"
echo
echo "Checking cuSPARSELt library..."
for candidate in \
  "$LIB_DEST/libcusparseLt.so.0" \
  /usr/local/cuda/targets/aarch64-linux/lib/libcusparseLt.so.0 \
  /usr/local/cuda/lib64/libcusparseLt.so.0 \
  /usr/lib/aarch64-linux-gnu/libcusparseLt.so.0
do
  if [[ -f "$candidate" ]]; then
    echo "Found: $candidate"
    exit 0
  fi
done

echo "cuSPARSELt installer completed, but libcusparseLt.so.0 was not found in the usual locations." >&2
exit 1

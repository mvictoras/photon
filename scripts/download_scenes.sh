#!/usr/bin/env bash
set -euo pipefail

SCENES_DIR="${1:-scenes}"
mkdir -p "$SCENES_DIR"

BITTERLI_BASE="https://benedikt-bitterli.me/resources/pbrt-v4"
SCENES=(cornell-box spaceship bathroom living-room staircase veach-mis kitchen)

echo "Downloading pbrt-v4 scenes to $SCENES_DIR..."
for scene in "${SCENES[@]}"; do
  if [ -d "$SCENES_DIR/$scene" ]; then
    echo "  $scene: already exists, skipping"
    continue
  fi
  echo "  Downloading $scene..."
  curl -L -o "$SCENES_DIR/${scene}.zip" "$BITTERLI_BASE/${scene}.zip"
  unzip -qo "$SCENES_DIR/${scene}.zip" -d "$SCENES_DIR"
  rm -f "$SCENES_DIR/${scene}.zip"
  echo "  $scene: done"
done

echo ""
echo "All scenes downloaded to $SCENES_DIR/"
echo "Available scenes:"
ls -d "$SCENES_DIR"/*/

#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
SCENES_DIR="${SCENES_DIR:-scenes}"
RENDERS_DIR="${RENDERS_DIR:-renders}"
SPP="${SPP:-512}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1920}"
EXPOSURE="${EXPOSURE:-0.3}"

RENDERER="$BUILD_DIR/app/photon_pbrt_render"

if [ ! -f "$RENDERER" ]; then
  echo "Error: renderer not found at $RENDERER"
  echo "Run scripts/build.sh first"
  exit 1
fi

mkdir -p "$RENDERS_DIR"

SCENES=(cornell-box spaceship bathroom living-room staircase veach-mis kitchen)

for scene in "${SCENES[@]}"; do
  SCENE_FILE="$SCENES_DIR/$scene/scene-v4.pbrt"
  if [ ! -f "$SCENE_FILE" ]; then
    echo "Skipping $scene: scene file not found at $SCENE_FILE"
    echo "  Run scripts/download_scenes.sh first"
    continue
  fi

  OUTPUT="$RENDERS_DIR/pbrt_${scene}_production.ppm"
  OUTPUT_PNG="$RENDERS_DIR/pbrt_${scene}_production.png"

  echo "=== Rendering $scene ==="
  echo "  $WIDTH x $HEIGHT, $SPP spp, exposure $EXPOSURE"

  PHOTON_BACKEND=kokkos "$RENDERER" "$SCENE_FILE" \
    -o "$OUTPUT" \
    --spp "$SPP" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --denoise \
    --exposure "$EXPOSURE"

  python3 -c "from PIL import Image; Image.open('$OUTPUT').save('$OUTPUT_PNG')" 2>/dev/null \
    && echo "  Saved: $OUTPUT_PNG" \
    || echo "  PPM saved: $OUTPUT (install Pillow for PNG conversion)"

  echo ""
done

echo "All renders saved to $RENDERS_DIR/"

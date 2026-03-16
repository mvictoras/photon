#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
KOKKOS_DIR="${KOKKOS_DIR:-}"
EMBREE_DIR="${EMBREE_DIR:-}"
OPTIX_DIR="${OPTIX_DIR:-}"

CMAKE_ARGS=(
  -S . -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
)

if [ -n "$KOKKOS_DIR" ]; then
  CMAKE_ARGS+=(-DKokkos_DIR="$KOKKOS_DIR")
fi

if [ -n "$EMBREE_DIR" ]; then
  CMAKE_ARGS+=(-DPHOTON_ENABLE_EMBREE=ON -Dembree_DIR="$EMBREE_DIR")
fi

if [ -n "$OPTIX_DIR" ]; then
  CMAKE_ARGS+=(-DPHOTON_ENABLE_OPTIX=ON -DOptiX_INSTALL_DIR="$OPTIX_DIR")
fi

CMAKE_ARGS+=(-DPHOTON_ENABLE_OIDN=ON)

echo "Configuring..."
cmake "${CMAKE_ARGS[@]}"

echo "Building..."
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "Build complete: $BUILD_DIR/"

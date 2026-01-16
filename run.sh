#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

if [[ "$(uname)" == "Darwin" ]]; then
  export DYLD_LIBRARY_PATH="build/src:build/_deps/anari-build:${DYLD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="build/src:build/_deps/anari-build:${LD_LIBRARY_PATH:-}"
fi

./build/app/photon_anari_render

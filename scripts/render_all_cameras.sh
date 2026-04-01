#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/home/mvictoras/src/photon/build:/home/mvictoras/src/embree-install/lib:/home/mvictoras/src/kokkos-cuda-install/lib:/opt/hfs21.0.596/dsolib:${LD_LIBRARY_PATH:-}

BINARY="./build/_deps/icet-build/bin/photon_pbrt_render"
SCENE_DIR="/storage/moana/island/pbrt-v4"
SCENE="$SCENE_DIR/island.pbrt"
OUTDIR="renders"
SPP=128
WIDTH=1920
HEIGHT=1080

# Extract scene body (from "Sampler" onwards — everything after the camera block)
SCENE_BODY="$(tail -n +77 "$SCENE")"

# Helper: write a per-camera pbrt to the scene dir (so relative includes resolve)
# and render it.
render_cam() {
  local name="$1"
  local scale="$2"       # e.g. "Scale -1 1 1" or ""
  local eye="$3"
  local at="$4"
  local up="$5"
  local fov="$6"
  local lens="$7"
  local focus="$8"

  local pbrt="$SCENE_DIR/_photon_${name}.pbrt"
  local out="$OUTDIR/moana_${name}.png"

  echo ""
  echo "=== Rendering $name → $out ==="

  {
    [ -n "$scale" ] && echo "$scale"
    cat <<HEADER
Film "rgb"
    "integer xresolution" [ $WIDTH ]
    "integer yresolution" [ $HEIGHT ]
    "float maxcomponentvalue" [ 10 ]
LookAt $eye
       $at
       $up
Camera "perspective"
    "float fov" [ $fov ]
    "float lensradius" [ $lens ]
    "float focaldistance" [ $focus ]
HEADER
    echo "$SCENE_BODY"
  } > "$pbrt"

  PHOTON_BACKEND=optix "$BINARY" "$pbrt" \
    -o "$out" \
    --spp $SPP \
    --width $WIDTH \
    --height $HEIGHT \
    --denoise \
    --use-ias --max-instances 50 --max-triangles 5000000

  rm -f "$pbrt"
  echo "Done: $out"
}

render_cam shotCam \
  "Scale -1 1 1" \
  "-1112.237781 23.286733 1425.865416" \
  "248.056442 238.807148 471.973941" \
  "-0.105327 0.991691 0.073859" \
  66.552369 0.003304 1675.338285

render_cam beachCam \
  "Scale -1 1 1" \
  "-510.523907 87.308744 181.770197" \
  "152.465305 30.939795 -72.727517" \
  "0.073871 0.996865 -0.028356" \
  54.432223 0.003125 712.391212

render_cam palmsCam \
  "Scale -1 1 1" \
  "-124.025465 405.622146 369.173046" \
  "472.312924 571.462848 16.499506" \
  "-0.200376 0.972526 0.118502" \
  54.432223 0.003125 712.391212

render_cam dunesACam \
  "Scale -1 1 1" \
  "-71.335795 78.734578 108.929948" \
  "-271.104819 80.800854 -297.054315" \
  "0.002016 0.999990 0.004097" \
  54.432223 0.003125 452.476689

render_cam hibiscusCam \
  "Scale -1 1 1" \
  "3.552661 850.641890 747.549775" \
  "237.075317 52.971848 -263.947975" \
  "0.137061 0.792946 -0.593676" \
  54.432223 0.003125 1309.174559

render_cam rootsCam \
  "Scale -1 1 1" \
  "-53.247680 63.459327 -57.577743" \
  "-44.095926 56.992385 -68.562019" \
  "0.263805 0.911128 -0.316628" \
  54.432223 0.003125 15.691725

render_cam grassCam \
  "Scale -1 1 1" \
  "-5.171249 20.334400 -89.973061" \
  "18.549567 8.008263 -107.847977" \
  "0.306118 0.923623 -0.230677" \
  54.432223 0.003125 32.157789

echo ""
echo "All cameras rendered. Output in $OUTDIR/"

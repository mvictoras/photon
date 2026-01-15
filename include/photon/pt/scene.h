#pragma once

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_mesh.h"

namespace photon::pt {

struct Scene
{
  TriangleMesh mesh;
  Bvh bvh;
};

} // namespace photon::pt

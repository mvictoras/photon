#pragma once

#include "opencode/pt/bvh/bvh.h"
#include "opencode/pt/geom/triangle_mesh.h"

namespace opencode::pt {

struct Scene
{
  TriangleMesh mesh;
  Bvh bvh;
};

} // namespace opencode::pt

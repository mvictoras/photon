#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace photon::pbrt {

struct PbrtVec3 {
  float x{0}, y{0}, z{0};
};

struct PbrtMaterial {
  std::string name;
  std::string type;
  PbrtVec3 reflectance{0.5f, 0.5f, 0.5f};
  PbrtVec3 eta{0.f, 0.f, 0.f};
  PbrtVec3 k{0.f, 0.f, 0.f};
  float roughness{0.5f};
  float uroughness{-1.f};
  float vroughness{-1.f};
  float eta_scalar{1.5f};
};

struct PbrtTriMesh {
  std::vector<float> positions;
  std::vector<int> indices;
  std::vector<float> normals;
  std::vector<float> uvs;
  std::string material_name;
  PbrtVec3 emission{0, 0, 0};
  bool is_emissive{false};
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

struct PbrtCamera {
  std::string type{"perspective"};
  float fov{45.f};
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

struct PbrtScene {
  int width{512};
  int height{512};
  int spp{64};
  int max_depth{8};

  PbrtCamera camera;
  std::map<std::string, PbrtMaterial> named_materials;
  std::vector<PbrtTriMesh> meshes;
};

}

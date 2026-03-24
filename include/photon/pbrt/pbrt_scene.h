#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace photon::pbrt {

struct PbrtVec3 {
  float x{0}, y{0}, z{0};
};

struct PbrtTexture {
  std::string name;
  std::string value_type;
  std::string class_type;
  std::string filename;
  std::vector<float> data;
  int width{0};
  int height{0};
  int channels{0};
};

struct PbrtMaterial {
  std::string name;
  std::string type;
  PbrtVec3 reflectance{0.5f, 0.5f, 0.5f};
  PbrtVec3 eta{0.f, 0.f, 0.f};
  PbrtVec3 k{0.f, 0.f, 0.f};
  float roughness{0.f};
  float uroughness{-1.f};
  float vroughness{-1.f};
  float eta_scalar{1.5f};
  std::string reflectance_texture;
};

struct PbrtTriMesh {
  std::vector<float> positions;
  std::vector<int> indices;
  std::vector<float> normals;
  std::vector<float> uvs;
  std::string material_name;
  std::string alpha_texture;
  PbrtVec3 emission{0, 0, 0};
  bool is_emissive{false};
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

struct PbrtCamera {
  std::string type{"perspective"};
  float fov{45.f};
  float lensradius{0.f};
  float focaldistance{0.f};
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  PbrtVec3 look_from{0,0,0};
  PbrtVec3 look_at_pt{0,0,1};
  PbrtVec3 look_up{0,1,0};
  float scale[3]{1,1,1};
  bool has_lookat{false};
};

struct PbrtInstance {
  std::string object_name;
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

struct PbrtScene {
  int width{512};
  int height{512};
  int spp{64};
  int max_depth{8};

  PbrtCamera camera;
  std::map<std::string, PbrtTexture> textures;
  std::map<std::string, PbrtMaterial> named_materials;
  std::vector<PbrtTriMesh> meshes;

  std::map<std::string, std::vector<PbrtTriMesh>> object_defs;
  std::vector<PbrtInstance> object_instances;

  std::string env_map_filename;
  float env_map_transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  float env_map_scale{1.f};
  bool has_env_map{false};

  uint64_t max_instances_per_object{500};
  uint64_t max_total_triangles{500000000};
  bool use_instancing{false};
};

}

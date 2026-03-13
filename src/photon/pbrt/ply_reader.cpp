#include "photon/pbrt/ply_reader.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace photon::pbrt {

namespace {

enum class PlyFormat { ASCII, BINARY_LITTLE_ENDIAN, BINARY_BIG_ENDIAN };

struct PlyProperty {
  std::string name;
  std::string type;
  bool is_list{false};
  std::string count_type;
  std::string value_type;
};

struct PlyElement {
  std::string name;
  int count{0};
  std::vector<PlyProperty> properties;
};

int type_size(const std::string &t)
{
  if (t == "float" || t == "float32" || t == "int" || t == "int32" || t == "uint" || t == "uint32") return 4;
  if (t == "double" || t == "float64" || t == "int64" || t == "uint64") return 8;
  if (t == "uchar" || t == "uint8" || t == "char" || t == "int8") return 1;
  if (t == "short" || t == "int16" || t == "ushort" || t == "uint16") return 2;
  return 4;
}

float read_float_bin(std::ifstream &in) {
  float v; in.read(reinterpret_cast<char*>(&v), 4); return v;
}

int read_int_bin(std::ifstream &in, const std::string &type) {
  if (type == "uchar" || type == "uint8") {
    uint8_t v; in.read(reinterpret_cast<char*>(&v), 1); return int(v);
  } else if (type == "int" || type == "int32") {
    int32_t v; in.read(reinterpret_cast<char*>(&v), 4); return int(v);
  } else if (type == "uint" || type == "uint32") {
    uint32_t v; in.read(reinterpret_cast<char*>(&v), 4); return int(v);
  } else if (type == "short" || type == "int16") {
    int16_t v; in.read(reinterpret_cast<char*>(&v), 2); return int(v);
  } else if (type == "ushort" || type == "uint16") {
    uint16_t v; in.read(reinterpret_cast<char*>(&v), 2); return int(v);
  }
  int32_t v; in.read(reinterpret_cast<char*>(&v), 4); return int(v);
}

void skip_bytes(std::ifstream &in, int n) {
  in.seekg(n, std::ios::cur);
}

}

bool load_ply(const std::string &path, PbrtTriMesh &mesh)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::fprintf(stderr, "PLY: cannot open %s\n", path.c_str());
    return false;
  }

  std::string line;
  std::getline(in, line);
  if (line.find("ply") == std::string::npos) {
    std::fprintf(stderr, "PLY: not a PLY file: %s\n", path.c_str());
    return false;
  }

  PlyFormat format = PlyFormat::BINARY_LITTLE_ENDIAN;
  std::vector<PlyElement> elements;

  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '\r')
      continue;
    if (line.back() == '\r')
      line.pop_back();

    std::istringstream iss(line);
    std::string keyword;
    iss >> keyword;

    if (keyword == "end_header")
      break;

    if (keyword == "format") {
      std::string fmt;
      iss >> fmt;
      if (fmt == "ascii")
        format = PlyFormat::ASCII;
      else if (fmt == "binary_big_endian")
        format = PlyFormat::BINARY_BIG_ENDIAN;
    } else if (keyword == "element") {
      PlyElement e;
      iss >> e.name >> e.count;
      elements.push_back(e);
    } else if (keyword == "property") {
      if (elements.empty()) continue;
      PlyProperty p;
      std::string w;
      iss >> w;
      if (w == "list") {
        p.is_list = true;
        iss >> p.count_type >> p.value_type >> p.name;
      } else {
        p.type = w;
        iss >> p.name;
      }
      elements.back().properties.push_back(p);
    }
  }

  if (format == PlyFormat::BINARY_BIG_ENDIAN) {
    std::fprintf(stderr, "PLY: big-endian not supported: %s\n", path.c_str());
    return false;
  }

  int vertex_idx = -1, face_idx = -1;
  for (int i = 0; i < int(elements.size()); ++i) {
    if (elements[i].name == "vertex") vertex_idx = i;
    if (elements[i].name == "face") face_idx = i;
  }

  if (vertex_idx < 0 || face_idx < 0) {
    std::fprintf(stderr, "PLY: missing vertex or face element: %s\n", path.c_str());
    return false;
  }

  const auto &velem = elements[vertex_idx];
  const auto &felem = elements[face_idx];

  int px = -1, py = -1, pz = -1, nx = -1, ny = -1, nz = -1, pu = -1, pv = -1;
  for (int i = 0; i < int(velem.properties.size()); ++i) {
    const auto &name = velem.properties[i].name;
    if (name == "x") px = i;
    else if (name == "y") py = i;
    else if (name == "z") pz = i;
    else if (name == "nx") nx = i;
    else if (name == "ny") ny = i;
    else if (name == "nz") nz = i;
    else if (name == "u" || name == "s" || name == "texture_u") pu = i;
    else if (name == "v" || name == "t" || name == "texture_v") pv = i;
  }

  bool has_normals = (nx >= 0 && ny >= 0 && nz >= 0);
  bool has_uvs = (pu >= 0 && pv >= 0);

  if (format == PlyFormat::BINARY_LITTLE_ENDIAN) {
    mesh.positions.resize(velem.count * 3);
    if (has_normals) mesh.normals.resize(velem.count * 3);
    if (has_uvs) mesh.uvs.resize(velem.count * 2);

    for (int vi = 0; vi < velem.count; ++vi) {
      float vals[16]{};
      int nprops = int(velem.properties.size());
      for (int pi = 0; pi < nprops; ++pi) {
        const auto &prop = velem.properties[pi];
        if (prop.type == "float" || prop.type == "float32")
          vals[pi] = read_float_bin(in);
        else
          skip_bytes(in, type_size(prop.type));
      }

      if (px >= 0) mesh.positions[vi*3+0] = vals[px];
      if (py >= 0) mesh.positions[vi*3+1] = vals[py];
      if (pz >= 0) mesh.positions[vi*3+2] = vals[pz];
      if (has_normals) {
        mesh.normals[vi*3+0] = vals[nx];
        mesh.normals[vi*3+1] = vals[ny];
        mesh.normals[vi*3+2] = vals[nz];
      }
      if (has_uvs) {
        mesh.uvs[vi*2+0] = vals[pu];
        mesh.uvs[vi*2+1] = vals[pv];
      }
    }

    for (int fi = 0; fi < felem.count; ++fi) {
      for (int pi = 0; pi < int(felem.properties.size()); ++pi) {
        const auto &prop = felem.properties[pi];
        if (prop.is_list && (prop.name == "vertex_indices" || prop.name == "vertex_index")) {
          int count = read_int_bin(in, prop.count_type);
          std::vector<int> face_indices(count);
          for (int j = 0; j < count; ++j)
            face_indices[j] = read_int_bin(in, prop.value_type);

          for (int j = 1; j < count - 1; ++j) {
            mesh.indices.push_back(face_indices[0]);
            mesh.indices.push_back(face_indices[j]);
            mesh.indices.push_back(face_indices[j + 1]);
          }
        } else if (prop.is_list) {
          int count = read_int_bin(in, prop.count_type);
          for (int j = 0; j < count; ++j)
            skip_bytes(in, type_size(prop.value_type));
        } else {
          skip_bytes(in, type_size(prop.type));
        }
      }
    }
  } else {
    mesh.positions.resize(velem.count * 3);
    if (has_normals) mesh.normals.resize(velem.count * 3);
    if (has_uvs) mesh.uvs.resize(velem.count * 2);

    for (int vi = 0; vi < velem.count; ++vi) {
      float vals[16]{};
      for (int pi = 0; pi < int(velem.properties.size()); ++pi) {
        float v;
        in >> v;
        vals[pi] = v;
      }
      if (px >= 0) mesh.positions[vi*3+0] = vals[px];
      if (py >= 0) mesh.positions[vi*3+1] = vals[py];
      if (pz >= 0) mesh.positions[vi*3+2] = vals[pz];
      if (has_normals) {
        mesh.normals[vi*3+0] = vals[nx];
        mesh.normals[vi*3+1] = vals[ny];
        mesh.normals[vi*3+2] = vals[nz];
      }
      if (has_uvs) {
        mesh.uvs[vi*2+0] = vals[pu];
        mesh.uvs[vi*2+1] = vals[pv];
      }
    }

    for (int fi = 0; fi < felem.count; ++fi) {
      int count;
      in >> count;
      std::vector<int> face_indices(count);
      for (int j = 0; j < count; ++j)
        in >> face_indices[j];
      for (int j = 1; j < count - 1; ++j) {
        mesh.indices.push_back(face_indices[0]);
        mesh.indices.push_back(face_indices[j]);
        mesh.indices.push_back(face_indices[j + 1]);
      }
    }
  }

  return !mesh.positions.empty() && !mesh.indices.empty();
}

}

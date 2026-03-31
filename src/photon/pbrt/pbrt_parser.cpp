#include "photon/pbrt/pbrt_parser.h"
#include "photon/pbrt/ply_reader.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stack>
#include <stdexcept>

namespace photon::pbrt {

namespace {

struct Token {
  enum Kind { STRING, NUMBER, LBRACKET, RBRACKET, IDENTIFIER, END };
  Kind kind{END};
  std::string text;
  float num{0.f};
};

struct Tokenizer {
  const char *p;
  const char *end;

  explicit Tokenizer(const std::string &src)
      : p(src.data()), end(src.data() + src.size())
  {}

  void skip_whitespace_and_comments()
  {
    while (p < end) {
      if (std::isspace(static_cast<unsigned char>(*p))) {
        ++p;
      } else if (*p == '#') {
        while (p < end && *p != '\n')
          ++p;
      } else {
        break;
      }
    }
  }

  Token next()
  {
    skip_whitespace_and_comments();
    if (p >= end)
      return {Token::END, "", 0.f};

    if (*p == '[') {
      ++p;
      return {Token::LBRACKET, "[", 0.f};
    }
    if (*p == ']') {
      ++p;
      return {Token::RBRACKET, "]", 0.f};
    }
    if (*p == '"') {
      ++p;
      const char *start = p;
      while (p < end && *p != '"')
        ++p;
      std::string s(start, p);
      if (p < end)
        ++p;
      return {Token::STRING, std::move(s), 0.f};
    }
    if (*p == '-' || *p == '+' || *p == '.' || std::isdigit(static_cast<unsigned char>(*p))) {
      const char *start = p;
      if (*p == '-' || *p == '+')
        ++p;
      while (p < end && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+'))
        ++p;
      std::string s(start, p);
      return {Token::NUMBER, s, std::strtof(s.c_str(), nullptr)};
    }
    if (std::isalpha(static_cast<unsigned char>(*p)) || *p == '_') {
      const char *start = p;
      while (p < end && (std::isalnum(static_cast<unsigned char>(*p)) || *p == '_'))
        ++p;
      return {Token::IDENTIFIER, std::string(start, p), 0.f};
    }

    ++p;
    return next();
  }

  Token peek()
  {
    const char *saved = p;
    Token t = next();
    p = saved;
    return t;
  }
};

struct ParamValue {
  std::string type;
  std::string name;
  std::vector<float> floats;
  std::vector<int> ints;
  std::vector<std::string> strings;
};

ParamValue parse_param(Tokenizer &tok)
{
  Token t = tok.next();
  if (t.kind != Token::STRING)
    return {};

  ParamValue pv;
  auto space = t.text.find(' ');
  if (space != std::string::npos) {
    pv.type = t.text.substr(0, space);
    pv.name = t.text.substr(space + 1);
  } else {
    pv.name = t.text;
  }

  Token bracket = tok.peek();
  if (bracket.kind == Token::STRING && (pv.type == "string" || pv.type == "texture")) {
    tok.next();
    pv.strings.push_back(bracket.text);
    return pv;
  }
  if (bracket.kind == Token::NUMBER && (pv.type == "float" || pv.type == "integer" || pv.type == "bool")) {
    tok.next();
    if (pv.type == "integer" || pv.type == "bool")
      pv.ints.push_back(int(bracket.num));
    else
      pv.floats.push_back(bracket.num);
    return pv;
  }
  if (bracket.kind != Token::LBRACKET)
    return pv;

  tok.next();

  while (true) {
    Token v = tok.peek();
    if (v.kind == Token::RBRACKET) {
      tok.next();
      break;
    }
    if (v.kind == Token::END)
      break;
    if (v.kind == Token::NUMBER) {
      tok.next();
      if (pv.type == "integer" || pv.type == "bool")
        pv.ints.push_back(int(v.num));
      else
        pv.floats.push_back(v.num);
    } else if (v.kind == Token::STRING) {
      tok.next();
      pv.strings.push_back(v.text);
    } else {
      tok.next();
    }
  }

  return pv;
}

struct ParseState {
  std::string current_material;
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  PbrtVec3 area_light_emission{0,0,0};
  bool in_area_light{false};
};

// sRGB approximations of measured spectral data for common metals
PbrtVec3 lookup_named_spectrum(const std::string &name) {
  if (name == "metal-Ag-eta") return {0.155f, 0.116f, 0.138f};
  if (name == "metal-Ag-k")   return {4.828f, 3.122f, 2.146f};
  if (name == "metal-Au-eta") return {0.143f, 0.374f, 1.442f};
  if (name == "metal-Au-k")   return {3.983f, 2.387f, 1.603f};
  if (name == "metal-Cu-eta") return {0.200f, 0.924f, 1.102f};
  if (name == "metal-Cu-k")   return {3.912f, 2.452f, 2.142f};
  if (name == "metal-Al-eta") return {1.345f, 0.965f, 0.612f};
  if (name == "metal-Al-k")   return {7.474f, 6.400f, 5.275f};
  if (name == "metal-Fe-eta") return {2.912f, 2.950f, 2.584f};
  if (name == "metal-Fe-k")   return {3.070f, 2.930f, 2.771f};
  return {0.f, 0.f, 0.f};
}

}

static std::string resolve_dir(const std::string &path)
{
  auto s = path.find_last_of("/\\");
  return (s != std::string::npos) ? path.substr(0, s + 1) : "";
}

static std::string g_root_dir;

static std::string preprocess_includes(const std::string &path, int depth = 0)
{
  if (depth > 10)
    return "";
  std::ifstream f(path);
  if (!f.is_open()) {
    std::fprintf(stderr, "  Warning: cannot open %s\n", path.c_str());
    return "";
  }
  std::string result;
  std::string line;
  while (std::getline(f, line)) {
    std::string trimmed = line;
    size_t start = trimmed.find_first_not_of(" \t");
    if (start != std::string::npos)
      trimmed = trimmed.substr(start);
    if (trimmed.size() > 5 && (trimmed.compare(0, 7, "Include") == 0 || trimmed.compare(0, 6, "Import") == 0)) {
      auto q1 = trimmed.find('"');
      auto q2 = trimmed.rfind('"');
      if (q1 != std::string::npos && q2 > q1) {
        std::string rel = trimmed.substr(q1 + 1, q2 - q1 - 1);
        std::string inc_file = g_root_dir + rel;
        std::fprintf(stderr, "  Including: %s\n", inc_file.c_str());
        result += preprocess_includes(inc_file, depth + 1);
        continue;
      }
    }
    result += line;
    result += '\n';
  }
  return result;
}

PbrtScene parse_pbrt_file(const std::string &path, bool use_instancing)
{
  std::string base_dir = resolve_dir(path);
  g_root_dir = base_dir;
  std::string content = preprocess_includes(path);

  Tokenizer tok(content);
  PbrtScene scene;
  scene.use_instancing = use_instancing;

  std::stack<ParseState> state_stack;
  ParseState state;

  bool in_world = false;
  std::map<std::string, std::vector<PbrtTriMesh>> object_defs;
  std::string current_object;
  std::map<std::string, uint64_t> instance_counts;
  uint64_t total_triangles = 0;

  auto add_mesh = [&](PbrtTriMesh &&m) {
    if (!current_object.empty())
      object_defs[current_object].push_back(std::move(m));
    else {
      total_triangles += m.indices.size() / 3;
      scene.meshes.push_back(std::move(m));
    }
  };

  for (int i = 0; i < 16; ++i)
    state.transform[i] = (i % 5 == 0) ? 1.f : 0.f;

  while (true) {
    Token t = tok.next();
    if (t.kind == Token::END)
      break;

    if (t.kind != Token::IDENTIFIER)
      continue;

    if (t.text == "Integrator") {
      Token type = tok.next();
      while (tok.peek().kind == Token::STRING) {
        auto pv = parse_param(tok);
        if (pv.name == "maxdepth" && !pv.ints.empty())
          scene.max_depth = pv.ints[0];
      }
    } else if (t.text == "Sampler") {
      Token type = tok.next();
      while (tok.peek().kind == Token::STRING) {
        auto pv = parse_param(tok);
        if (pv.name == "pixelsamples" && !pv.ints.empty())
          scene.spp = pv.ints[0];
      }
    } else if (t.text == "Film") {
      Token type = tok.next();
      while (tok.peek().kind == Token::STRING) {
        auto pv = parse_param(tok);
        if (pv.name == "xresolution" && !pv.ints.empty())
          scene.width = pv.ints[0];
        else if (pv.name == "yresolution" && !pv.ints.empty())
          scene.height = pv.ints[0];
      }
    } else if (t.text == "PixelFilter") {
      Token type = tok.next();
      while (tok.peek().kind == Token::STRING)
        parse_param(tok);
    } else if (t.text == "Camera") {
      Token type = tok.next();
      scene.camera.type = type.text;
      std::memcpy(scene.camera.transform, state.transform, 16 * sizeof(float));
      while (tok.peek().kind == Token::STRING) {
        auto pv = parse_param(tok);
        if (pv.name == "fov" && !pv.floats.empty())
          scene.camera.fov = pv.floats[0];
        else if (pv.name == "lensradius" && !pv.floats.empty())
          scene.camera.lensradius = pv.floats[0];
        else if (pv.name == "focaldistance" && !pv.floats.empty())
          scene.camera.focaldistance = pv.floats[0];
      }
    } else if (t.text == "Transform") {
      Token bracket = tok.next();
      if (bracket.kind == Token::LBRACKET) {
        for (int i = 0; i < 16; ++i) {
          Token v = tok.next();
          state.transform[i] = v.num;
        }
        tok.next();
      }
    } else if (t.text == "LookAt") {
      float vals[9];
      for (int i = 0; i < 9; ++i) {
        Token v = tok.next();
        vals[i] = v.num;
      }
      scene.camera.look_from = {vals[0], vals[1], vals[2]};
      scene.camera.look_at_pt = {vals[3], vals[4], vals[5]};
      scene.camera.look_up = {vals[6], vals[7], vals[8]};
      scene.camera.has_lookat = true;
    } else if (t.text == "Scale") {
      float sx = tok.next().num, sy = tok.next().num, sz = tok.next().num;
      float scale_m[16]{sx,0,0,0, 0,sy,0,0, 0,0,sz,0, 0,0,0,1};
      float result[16];
      auto mul = [](const float *a, const float *b, float *r) {
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            float s = 0;
            for (int k = 0; k < 4; ++k)
              s += a[i*4+k] * b[k*4+j];
            r[i*4+j] = s;
          }
      };
      mul(state.transform, scale_m, result);
      std::memcpy(state.transform, result, 64);
      if (!in_world) {
        scene.camera.scale[0] *= sx;
        scene.camera.scale[1] *= sy;
        scene.camera.scale[2] *= sz;
      }
    } else if (t.text == "Rotate") {
      float angle = tok.next().num;
      float ax = tok.next().num, ay = tok.next().num, az = tok.next().num;
      float len = std::sqrt(ax*ax + ay*ay + az*az);
      if (len > 0.f) { ax /= len; ay /= len; az /= len; }
      float rad = angle * 3.14159265f / 180.f;
      float c = std::cos(rad), s = std::sin(rad), t2 = 1.f - c;
      float rot[16]{
        t2*ax*ax+c, t2*ax*ay-s*az, t2*ax*az+s*ay, 0,
        t2*ax*ay+s*az, t2*ay*ay+c, t2*ay*az-s*ax, 0,
        t2*ax*az-s*ay, t2*ay*az+s*ax, t2*az*az+c, 0,
        0,0,0,1};
      float result[16];
      auto mul = [](const float *a, const float *b, float *r) {
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            float sv = 0;
            for (int k = 0; k < 4; ++k)
              sv += a[i*4+k] * b[k*4+j];
            r[i*4+j] = sv;
          }
      };
      mul(state.transform, rot, result);
      std::memcpy(state.transform, result, 64);
    } else if (t.text == "ConcatTransform") {
      Token bracket = tok.next();
      if (bracket.kind == Token::LBRACKET) {
        float concat[16];
        for (int i = 0; i < 16; ++i)
          concat[i] = tok.next().num;
        tok.next();
        float result[16];
        auto mul = [](const float *a, const float *b, float *r) {
          for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
              float sv = 0;
              for (int k = 0; k < 4; ++k)
                sv += a[i*4+k] * b[k*4+j];
              r[i*4+j] = sv;
            }
        };
        // pbrt-v4 post-multiplies: new_CTM = old_CTM * M_concat
        mul(concat, state.transform, result);
        std::memcpy(state.transform, result, 64);
      }
    } else if (t.text == "Identity") {
      for (int i = 0; i < 16; ++i)
        state.transform[i] = (i % 5 == 0) ? 1.f : 0.f;
    } else if (t.text == "WorldBegin") {
      in_world = true;
      for (int i = 0; i < 16; ++i)
        state.transform[i] = (i % 5 == 0) ? 1.f : 0.f;
    } else if (t.text == "AttributeBegin") {
      state_stack.push(state);
    } else if (t.text == "AttributeEnd") {
      if (!state_stack.empty()) {
        state = state_stack.top();
        state_stack.pop();
      }
    } else if (t.text == "MakeNamedMaterial") {
      Token name = tok.next();
      PbrtMaterial mat;
      mat.name = name.text;
      while (tok.peek().kind == Token::STRING) {
        auto pv = parse_param(tok);
        if (pv.name == "type" && !pv.strings.empty())
          mat.type = pv.strings[0];
        else if (pv.name == "reflectance" && pv.type == "texture" && !pv.strings.empty())
          mat.reflectance_texture = pv.strings[0];
        else if (pv.name == "reflectance" && pv.floats.size() >= 3)
          mat.reflectance = {pv.floats[0], pv.floats[1], pv.floats[2]};
        else if (pv.name == "roughness" && !pv.floats.empty())
          mat.roughness = pv.floats[0];
        else if (pv.name == "eta") {
          if (!pv.strings.empty())
            mat.eta = lookup_named_spectrum(pv.strings[0]);
          else if (pv.floats.size() >= 3)
            mat.eta = {pv.floats[0], pv.floats[1], pv.floats[2]};
          else if (pv.floats.size() == 1)
            mat.eta_scalar = pv.floats[0];
        } else if (pv.name == "k") {
          if (!pv.strings.empty())
            mat.k = lookup_named_spectrum(pv.strings[0]);
          else if (pv.floats.size() >= 3)
            mat.k = {pv.floats[0], pv.floats[1], pv.floats[2]};
        }
        else if ((pv.name == "uroughness") && !pv.floats.empty())
          mat.uroughness = pv.floats[0];
        else if ((pv.name == "vroughness") && !pv.floats.empty())
          mat.vroughness = pv.floats[0];
      }
      scene.named_materials[mat.name] = mat;
    } else if (t.text == "NamedMaterial") {
      Token name = tok.next();
      state.current_material = name.text;
    } else if (t.text == "AreaLightSource") {
      Token type = tok.next();
      state.in_area_light = true;
      while (tok.peek().kind == Token::STRING) {
        auto pv = parse_param(tok);
        if (pv.name == "L" && pv.floats.size() >= 3)
          state.area_light_emission = {pv.floats[0], pv.floats[1], pv.floats[2]};
      }
    } else if (t.text == "Shape") {
      Token type = tok.next();
      if (type.text == "trianglemesh") {
        PbrtTriMesh mesh;
        mesh.material_name = state.current_material;
        std::memcpy(mesh.transform, state.transform, 16 * sizeof(float));
        if (state.in_area_light) {
          mesh.is_emissive = true;
          mesh.emission = state.area_light_emission;
        }
        while (tok.peek().kind == Token::STRING) {
          auto pv = parse_param(tok);
          if ((pv.name == "P" || pv.name == "p") && pv.type == "point3")
            mesh.positions = std::move(pv.floats);
          else if (pv.name == "indices" && pv.type == "integer")
            mesh.indices = std::move(pv.ints);
          else if (pv.name == "N" && pv.type == "normal")
            mesh.normals = std::move(pv.floats);
          else if (pv.name == "uv" && pv.type == "point2")
            mesh.uvs = std::move(pv.floats);
        }
        if (!mesh.positions.empty() && !mesh.indices.empty())
          add_mesh(std::move(mesh));
      } else if (type.text == "plymesh") {
        PbrtTriMesh mesh;
        mesh.material_name = state.current_material;
        std::memcpy(mesh.transform, state.transform, 16 * sizeof(float));
        if (state.in_area_light) {
          mesh.is_emissive = true;
          mesh.emission = state.area_light_emission;
        }
        std::string ply_filename;
        while (tok.peek().kind == Token::STRING) {
          auto pv = parse_param(tok);
          if (pv.name == "filename" && !pv.strings.empty())
            ply_filename = pv.strings[0];
          else if (pv.name == "alpha" && pv.type == "texture" && !pv.strings.empty())
            mesh.alpha_texture = pv.strings[0];
        }
        if (!ply_filename.empty()) {
          std::string ply_path = base_dir + ply_filename;
          if (load_ply(ply_path, mesh))
            add_mesh(std::move(mesh));
          else
            std::fprintf(stderr, "Warning: failed to load PLY: %s\n", ply_path.c_str());
        }
      } else if (type.text == "sphere") {
        PbrtTriMesh mesh;
        mesh.material_name = state.current_material;
        std::memcpy(mesh.transform, state.transform, 16 * sizeof(float));
        if (state.in_area_light) {
          mesh.is_emissive = true;
          mesh.emission = state.area_light_emission;
        }
        float radius = 1.f;
        while (tok.peek().kind == Token::STRING) {
          auto pv = parse_param(tok);
          if (pv.name == "radius" && !pv.floats.empty())
            radius = pv.floats[0];
        }

        const int slices = 32, stacks = 16;
        for (int j = 0; j <= stacks; ++j) {
          float v = float(j) / float(stacks);
          float theta = v * 3.14159265f;
          float sin_t = std::sin(theta), cos_t = std::cos(theta);
          for (int i = 0; i <= slices; ++i) {
            float u_f = float(i) / float(slices);
            float phi = u_f * 6.28318530f;
            float x = sin_t * std::cos(phi) * radius;
            float y = cos_t * radius;
            float z = sin_t * std::sin(phi) * radius;
            mesh.positions.push_back(x);
            mesh.positions.push_back(y);
            mesh.positions.push_back(z);
            mesh.normals.push_back(sin_t * std::cos(phi));
            mesh.normals.push_back(cos_t);
            mesh.normals.push_back(sin_t * std::sin(phi));
          }
        }
        for (int j = 0; j < stacks; ++j) {
          for (int i = 0; i < slices; ++i) {
            int a = j * (slices + 1) + i;
            int b = a + 1;
            int c = a + slices + 1;
            int d = c + 1;
            mesh.indices.push_back(a);
            mesh.indices.push_back(c);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);
            mesh.indices.push_back(d);
          }
        }
        if (!mesh.positions.empty() && !mesh.indices.empty())
          add_mesh(std::move(mesh));
      } else {
        while (tok.peek().kind == Token::STRING)
          parse_param(tok);
      }
    } else if (t.text == "Texture") {
      Token tex_name = tok.next();
      Token value_type = tok.next();
      Token class_type = tok.next();
      PbrtTexture tex;
      tex.name = tex_name.text;
      tex.value_type = value_type.text;
      tex.class_type = class_type.text;
      while (tok.peek().kind == Token::STRING) {
        auto pv = parse_param(tok);
        if (pv.name == "filename" && !pv.strings.empty())
          tex.filename = pv.strings[0];
      }
      scene.textures[tex.name] = tex;
    } else if (t.text == "LightSource") {
      Token type = tok.next();
      if (type.text == "infinite") {
        scene.has_env_map = true;
        std::memcpy(scene.env_map_transform, state.transform, 16 * sizeof(float));
        while (tok.peek().kind == Token::STRING) {
          auto pv = parse_param(tok);
          if (pv.name == "filename" && !pv.strings.empty())
            scene.env_map_filename = pv.strings[0];
          else if (pv.name == "scale" && pv.floats.size() >= 3)
            scene.env_map_scale = (pv.floats[0] + pv.floats[1] + pv.floats[2]) / 3.f;
          else if (pv.name == "L" && pv.floats.size() >= 3)
            scene.env_map_scale = (pv.floats[0] + pv.floats[1] + pv.floats[2]) / 3.f;
        }
      } else {
        while (tok.peek().kind == Token::STRING)
          parse_param(tok);
      }
    } else if (t.text == "ObjectBegin") {
      Token name = tok.next();
      current_object = name.text;
    } else if (t.text == "ObjectEnd") {
      current_object.clear();
    } else if (t.text == "ObjectInstance") {
      Token name = tok.next();
      auto it = object_defs.find(name.text);
      if (it == object_defs.end())
        continue;

      PbrtInstance inst;
      inst.object_name = name.text;
      std::memcpy(inst.transform, state.transform, 16 * sizeof(float));
      scene.object_instances.push_back(inst);

      if (scene.use_instancing) continue;

      if (total_triangles >= scene.max_total_triangles)
        continue;
      auto &ic = instance_counts[name.text];
      if (ic >= scene.max_instances_per_object)
        continue;
      uint64_t obj_tris = 0;
      for (const auto &obj_mesh : it->second)
        obj_tris += obj_mesh.indices.size() / 3;
      if (total_triangles + obj_tris > scene.max_total_triangles)
        continue;
      for (const auto &obj_mesh : it->second) {
        PbrtTriMesh m = obj_mesh;
        std::memcpy(m.transform, state.transform, 16 * sizeof(float));
        m.material_name = obj_mesh.material_name;
        scene.meshes.push_back(std::move(m));
      }
      total_triangles += obj_tris;
      ic++;
    } else if (t.text == "Material"
               || t.text == "ReverseOrientation"
               || t.text == "TransformBegin" || t.text == "TransformEnd"
               || t.text == "MediumInterface" || t.text == "MakeNamedMedium") {
      if (t.text == "Material"
          || t.text == "MakeNamedMedium") {
        tok.next();
      }
      while (tok.peek().kind == Token::STRING)
        parse_param(tok);
    }
  }

  scene.object_defs = std::move(object_defs);

  return scene;
}

}

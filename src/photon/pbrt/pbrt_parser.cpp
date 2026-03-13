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

}

PbrtScene parse_pbrt_file(const std::string &path)
{
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Cannot open pbrt file: " + path);

  std::string base_dir;
  auto last_slash = path.find_last_of("/\\");
  if (last_slash != std::string::npos)
    base_dir = path.substr(0, last_slash + 1);

  std::stringstream ss;
  ss << file.rdbuf();
  std::string content = ss.str();

  Tokenizer tok(content);
  PbrtScene scene;

  std::stack<ParseState> state_stack;
  ParseState state;

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
    } else if (t.text == "WorldBegin") {
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
        else if (pv.name == "reflectance" && pv.floats.size() >= 3)
          mat.reflectance = {pv.floats[0], pv.floats[1], pv.floats[2]};
        else if (pv.name == "roughness" && !pv.floats.empty())
          mat.roughness = pv.floats[0];
        else if (pv.name == "eta") {
          if (pv.floats.size() >= 3)
            mat.eta = {pv.floats[0], pv.floats[1], pv.floats[2]};
          else if (pv.floats.size() == 1)
            mat.eta_scalar = pv.floats[0];
        } else if (pv.name == "k" && pv.floats.size() >= 3)
          mat.k = {pv.floats[0], pv.floats[1], pv.floats[2]};
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
          scene.meshes.push_back(std::move(mesh));
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
        }
        if (!ply_filename.empty()) {
          std::string ply_path = base_dir + ply_filename;
          if (load_ply(ply_path, mesh))
            scene.meshes.push_back(std::move(mesh));
          else
            std::fprintf(stderr, "Warning: failed to load PLY: %s\n", ply_path.c_str());
        }
      } else {
        while (tok.peek().kind == Token::STRING)
          parse_param(tok);
      }
    } else if (t.text == "Texture" || t.text == "Material" || t.text == "LightSource"
               || t.text == "ConcatTransform" || t.text == "ReverseOrientation"
               || t.text == "TransformBegin" || t.text == "TransformEnd"
               || t.text == "ObjectBegin" || t.text == "ObjectEnd"
               || t.text == "ObjectInstance" || t.text == "Include"
               || t.text == "MediumInterface" || t.text == "MakeNamedMedium") {
      if (t.text == "Texture" || t.text == "Material" || t.text == "LightSource"
          || t.text == "MakeNamedMedium") {
        tok.next();
        if (t.text == "Texture")
          tok.next();
      }
      while (tok.peek().kind == Token::STRING)
        parse_param(tok);
    }
  }

  return scene;
}

}

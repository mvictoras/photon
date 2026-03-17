#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/camera.h"
#include "photon/pt/denoiser.h"
#include "photon/pt/light.h"
#include "photon/pt/material.h"
#include "photon/pt/math.h"
#include "photon/pt/pathtracer.h"
#include "photon/pt/scene.h"
#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_mesh.h"

namespace {

using namespace photon::pt;

static float aces_filmic(float x)
{
  const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  x = std::fmax(x, 0.f);
  return std::fmin((x * (a * x + b)) / (x * (c * x + d) + e), 1.f);
}

enum class ScalarFieldType { Gyroid, Turbulence, Supernova };

float noise3d(float x, float y, float z)
{
  float n = std::sin(x * 12.9898f + y * 78.233f + z * 45.164f);
  return (n - std::floor(n)) * 2.f - 1.f;
}

float scalar_field(float x, float y, float z, float freq, ScalarFieldType type = ScalarFieldType::Gyroid)
{
  switch (type) {
  case ScalarFieldType::Gyroid: {
    return std::sin(freq * x) * std::cos(freq * y)
         + std::sin(freq * y) * std::cos(freq * z)
         + std::sin(freq * z) * std::cos(freq * x);
  }
  case ScalarFieldType::Turbulence: {
    float v = 0.f, amp = 1.f, f = freq;
    for (int oct = 0; oct < 4; ++oct) {
      v += amp * (std::sin(f*x + 1.7f*std::cos(f*z*0.8f))
               * std::cos(f*y + 2.1f*std::sin(f*x*0.6f))
               + noise3d(f*x, f*y, f*z) * 0.3f);
      f *= 2.f;
      amp *= 0.5f;
    }
    return v;
  }
  case ScalarFieldType::Supernova: {
    float r = std::sqrt(x*x + y*y + z*z);
    float theta = std::atan2(std::sqrt(x*x + z*z), y);
    float phi = std::atan2(z, x);
    float shell = std::sin(freq * r) * std::exp(-r * 0.3f);
    float jet = std::exp(-5.f * (theta - 0.3f) * (theta - 0.3f)) * std::cos(freq * r * 0.5f)
              + std::exp(-5.f * (theta - 2.84f) * (theta - 2.84f)) * std::cos(freq * r * 0.5f);
    float spiral = std::sin(phi * 3.f + r * 2.f) * std::exp(-r * 0.5f) * 0.3f;
    return shell + jet * 0.8f + spiral;
  }
  }
  return 0.f;
}

// Marching cubes edge table (simplified — 12 edges per cube)
struct MCVertex {
  float x, y, z;
  float nx, ny, nz;
  float scalar;
};

void generate_isosurface(
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    int nx, int ny, int nz,
    float iso, float freq,
    std::vector<float> &positions,
    std::vector<float> &normals,
    std::vector<float> &colors,
    std::vector<int> &indices,
    ScalarFieldType field_type = ScalarFieldType::Gyroid)
{
  float dx = (x1 - x0) / nx;
  float dy = (y1 - y0) / ny;
  float dz = (z1 - z0) / nz;
  float eps = dx * 0.01f;

  auto interp = [&](float px, float py, float pz, float v0,
                     float qx, float qy, float qz, float v1,
                     MCVertex &out) {
    float t = (v0 != v1) ? (iso - v0) / (v1 - v0) : 0.5f;
    t = std::fmax(0.f, std::fmin(1.f, t));
    out.x = px + t * (qx - px);
    out.y = py + t * (qy - py);
    out.z = pz + t * (qz - pz);

    float gx = (scalar_field(out.x + eps, out.y, out.z, freq, field_type)
              - scalar_field(out.x - eps, out.y, out.z, freq, field_type)) / (2.f * eps);
    float gy = (scalar_field(out.x, out.y + eps, out.z, freq, field_type)
              - scalar_field(out.x, out.y - eps, out.z, freq, field_type)) / (2.f * eps);
    float gz = (scalar_field(out.x, out.y, out.z + eps, freq, field_type)
              - scalar_field(out.x, out.y, out.z - eps, freq, field_type)) / (2.f * eps);
    float len = std::sqrt(gx*gx + gy*gy + gz*gz);
    if (len > 0.f) { gx /= len; gy /= len; gz /= len; }
    out.nx = -gx; out.ny = -gy; out.nz = -gz;
    out.scalar = scalar_field(out.x, out.y, out.z, freq, field_type);
  };

  // Simple marching cubes — for each cell, check if isosurface crosses
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        float cx = x0 + ix * dx;
        float cy = y0 + iy * dy;
        float cz = z0 + iz * dz;

        // 8 corner values
        float v[8];
        v[0] = scalar_field(cx,    cy,    cz,    freq, field_type);
        v[1] = scalar_field(cx+dx, cy,    cz,    freq, field_type);
        v[2] = scalar_field(cx+dx, cy+dy, cz,    freq, field_type);
        v[3] = scalar_field(cx,    cy+dy, cz,    freq, field_type);
        v[4] = scalar_field(cx,    cy,    cz+dz, freq, field_type);
        v[5] = scalar_field(cx+dx, cy,    cz+dz, freq, field_type);
        v[6] = scalar_field(cx+dx, cy+dy, cz+dz, freq, field_type);
        v[7] = scalar_field(cx,    cy+dy, cz+dz, freq, field_type);

        // Simplified: for each of 6 tetrahedra in the cube, emit triangles
        int cube_index = 0;
        for (int i = 0; i < 8; ++i)
          if (v[i] < iso) cube_index |= (1 << i);

        if (cube_index == 0 || cube_index == 255) continue;

        // Corner positions
        float px[8] = {cx, cx+dx, cx+dx, cx, cx, cx+dx, cx+dx, cx};
        float py[8] = {cy, cy, cy+dy, cy+dy, cy, cy, cy+dy, cy+dy};
        float pz[8] = {cz, cz, cz, cz, cz+dz, cz+dz, cz+dz, cz+dz};

        // 12 edges of the cube
        static const int edge_pairs[12][2] = {
          {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
        };

        MCVertex edge_verts[12];
        int edge_valid[12] = {};

        for (int e = 0; e < 12; ++e) {
          int a = edge_pairs[e][0], b = edge_pairs[e][1];
          if ((cube_index & (1<<a)) != (cube_index & (1<<b))) {
            interp(px[a], py[a], pz[a], v[a], px[b], py[b], pz[b], v[b], edge_verts[e]);
            edge_valid[e] = 1;
          }
        }

        // Triangulation table (simplified — use a basic triangulation)
        // For each face of the cube, if the isosurface crosses it, emit a triangle
        static const int tri_table[256][16] = {
          #include "mc_table.inc"
        };

        for (int t = 0; tri_table[cube_index][t] != -1; t += 3) {
          int e0 = tri_table[cube_index][t];
          int e1 = tri_table[cube_index][t+1];
          int e2 = tri_table[cube_index][t+2];

          if (!edge_valid[e0] || !edge_valid[e1] || !edge_valid[e2]) continue;

          int base = int(positions.size() / 3);
          for (int vi : {e0, e1, e2}) {
            MCVertex &mv = edge_verts[vi];
            positions.push_back(mv.x);
            positions.push_back(mv.y);
            positions.push_back(mv.z);
            normals.push_back(mv.nx);
            normals.push_back(mv.ny);
            normals.push_back(mv.nz);
            // Color by scalar value (cool-warm colormap)
            float s = (mv.scalar + 2.f) * 0.25f;
            s = std::fmax(0.f, std::fmin(1.f, s));
            colors.push_back(s);
            colors.push_back(0.3f + 0.4f * (1.f - std::fabs(s - 0.5f) * 2.f));
            colors.push_back(1.f - s);
          }
          indices.push_back(base);
          indices.push_back(base + 1);
          indices.push_back(base + 2);
        }
      }
    }
  }
}

struct Pixel {
  float r, g, b, depth;
};

void depth_composite_op(void *invec, void *inoutvec, int *len, MPI_Datatype *)
{
  auto *in = static_cast<Pixel *>(invec);
  auto *out = static_cast<Pixel *>(inoutvec);
  for (int i = 0; i < *len; ++i) {
    if (in[i].depth > 0.f && (out[i].depth <= 0.f || in[i].depth < out[i].depth))
      out[i] = in[i];
  }
}

}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, num_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  Kokkos::initialize(argc, argv);
  {
    int grid_res = 128;
    int width = 1920, height = 1080;
    int spp = 256;
    float exposure = 0.5f;
    std::string output = "scivis_mpi.ppm";
    bool denoise = false;
    ScalarFieldType field_type = ScalarFieldType::Gyroid;
    std::string dataset_name = "gyroid";

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--res" && i+1 < argc) grid_res = std::atoi(argv[++i]);
      else if (arg == "--width" && i+1 < argc) width = std::atoi(argv[++i]);
      else if (arg == "--height" && i+1 < argc) height = std::atoi(argv[++i]);
      else if (arg == "--spp" && i+1 < argc) spp = std::atoi(argv[++i]);
      else if (arg == "--exposure" && i+1 < argc) exposure = std::atof(argv[++i]);
      else if (arg == "-o" && i+1 < argc) output = argv[++i];
      else if (arg == "--denoise") denoise = true;
      else if (arg == "--dataset" && i+1 < argc) {
        dataset_name = argv[++i];
        if (dataset_name == "gyroid") field_type = ScalarFieldType::Gyroid;
        else if (dataset_name == "turbulence") field_type = ScalarFieldType::Turbulence;
        else if (dataset_name == "supernova") field_type = ScalarFieldType::Supernova;
      }
    }

    if (rank == 0)
      std::fprintf(stderr, "Sci-vis MPI render: %d ranks, dataset=%s, grid %d³, %dx%d @ %d spp\n",
          num_ranks, dataset_name.c_str(), grid_res, width, height, spp);

    // Each rank generates isosurface for its Z-slab of the domain
    float domain_min = -3.14159f, domain_max = 3.14159f;
    float domain_size = domain_max - domain_min;
    float z_per_rank = domain_size / num_ranks;
    float my_z0 = domain_min + rank * z_per_rank;
    float my_z1 = my_z0 + z_per_rank;

    int nz_per_rank = grid_res / num_ranks;
    if (nz_per_rank < 1) nz_per_rank = 1;

    float iso_value = 0.0f;
    float field_freq = 2.0f;
    if (field_type == ScalarFieldType::Turbulence) {
      field_freq = 1.5f;
      iso_value = 0.3f;
    } else if (field_type == ScalarFieldType::Supernova) {
      field_freq = 3.0f;
      iso_value = 0.15f;
    }

    std::vector<float> positions, normals_vec, colors_vec;
    std::vector<int> indices_vec;

    auto t_mc0 = std::chrono::high_resolution_clock::now();
    generate_isosurface(domain_min, domain_min, my_z0,
                        domain_max, domain_max, my_z1,
                        grid_res, grid_res, nz_per_rank,
                        iso_value, field_freq,
                        positions, normals_vec, colors_vec, indices_vec,
                        field_type);
    auto t_mc1 = std::chrono::high_resolution_clock::now();
    double mc_ms = std::chrono::duration<double, std::milli>(t_mc1 - t_mc0).count();

    size_t num_verts = positions.size() / 3;
    size_t num_tris = indices_vec.size() / 3;

    std::fprintf(stderr, "Rank %d: z=[%.2f,%.2f], %zu tris, MC in %.0f ms\n",
        rank, my_z0, my_z1, num_tris, mc_ms);

    // Build photon scene
    TriangleMesh mesh;
    mesh.positions = Kokkos::View<Vec3 *>("pos", num_verts);
    mesh.indices = Kokkos::View<u32 *>("idx", num_verts);
    mesh.normals = Kokkos::View<Vec3 *>("nrm", num_verts);
    mesh.vertex_colors = Kokkos::View<Vec3 *>("vcol", num_verts);
    mesh.material_ids = Kokkos::View<u32 *>("mat", num_tris);
    mesh.albedo_per_prim = Kokkos::View<Vec3 *>("alb", num_tris);

    auto pos_h = Kokkos::create_mirror_view(mesh.positions);
    auto idx_h = Kokkos::create_mirror_view(mesh.indices);
    auto nrm_h = Kokkos::create_mirror_view(mesh.normals);
    auto vcol_h = Kokkos::create_mirror_view(mesh.vertex_colors);
    auto mat_h = Kokkos::create_mirror_view(mesh.material_ids);
    auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);

    for (size_t i = 0; i < num_verts; ++i) {
      pos_h(i) = {positions[i*3], positions[i*3+1], positions[i*3+2]};
      idx_h(i) = u32(i);
      nrm_h(i) = {normals_vec[i*3], normals_vec[i*3+1], normals_vec[i*3+2]};
      vcol_h(i) = {colors_vec[i*3], colors_vec[i*3+1], colors_vec[i*3+2]};
    }
    for (size_t i = 0; i < num_tris; ++i) {
      mat_h(i) = 0;
      alb_h(i) = {0.8f, 0.8f, 0.8f};
    }

    Kokkos::deep_copy(mesh.positions, pos_h);
    Kokkos::deep_copy(mesh.indices, idx_h);
    Kokkos::deep_copy(mesh.normals, nrm_h);
    Kokkos::deep_copy(mesh.vertex_colors, vcol_h);
    Kokkos::deep_copy(mesh.material_ids, mat_h);
    Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);

    Scene scene;
    scene.mesh = mesh;
    scene.bvh = Bvh::build_cpu(mesh);

    Material mat{};
    mat.base_color = {0.8f, 0.8f, 0.8f};
    mat.roughness = 0.3f;
    mat.metallic = 0.0f;
    scene.material_count = 1;
    scene.materials = Kokkos::View<Material *>("materials", 1);
    {
      Kokkos::View<Material *, Kokkos::HostSpace> mats_h("mh", 1);
      mats_h(0) = mat;
      Kokkos::deep_copy(scene.materials, mats_h);
    }

    // Area light above the isosurface
    Light light{};
    light.type = LightType::Area;
    light.position = {-3.f, 6.f, -3.f};
    light.edge1 = {6.f, 0.f, 0.f};
    light.edge2 = {0.f, 0.f, 6.f};
    light.area = 36.f;
    light.color = {8.f, 8.f, 8.f};
    light.intensity = 1.f;
    light.direction = {0.f, -1.f, 0.f};

    scene.light_count = 1;
    scene.lights = Kokkos::View<Light *>("lights", 1);
    {
      Kokkos::View<Light *, Kokkos::HostSpace> lh("lh", 1);
      lh(0) = light;
      Kokkos::deep_copy(scene.lights, lh);
    }

    float aspect = float(width) / float(height);
    Vec3 cam_pos, cam_target = {0.f, 0.f, 0.f};
    Vec3 cam_up = {0.f, 1.f, 0.f};
    float fov = 45.f;

    if (field_type == ScalarFieldType::Supernova) {
      cam_pos = {6.f, 3.f, 8.f};
      fov = 50.f;
    } else if (field_type == ScalarFieldType::Turbulence) {
      cam_pos = {0.f, 5.f, 9.f};
      fov = 50.f;
    } else {
      cam_pos = {0.f, 4.f, 10.f};
    }

    Camera camera = Camera::make_perspective(cam_pos, cam_target, cam_up, fov, aspect);

    auto backend = create_best_backend();
    backend->build_accel(scene);

    PathTracer pt;
    pt.params.width = u32(width);
    pt.params.height = u32(height);
    pt.params.samples_per_pixel = u32(spp);
    pt.params.max_depth = 8;
    pt.camera = camera;
    pt.set_scene(scene);
    pt.set_backend(std::move(backend));

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = pt.render();
    auto t1 = std::chrono::high_resolution_clock::now();
    double render_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "Rank %d: rendered in %.1f ms\n", rank, render_ms);

    // Composite
    size_t pixel_count = size_t(width) * height;
    auto color_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);
    auto depth_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.depth);

    std::vector<Pixel> local_pixels(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i) {
      u32 y = u32(i) / u32(width), x = u32(i) % u32(width);
      auto c = color_h(y, x);
      local_pixels[i] = {c.x, c.y, c.z, depth_h(y, x)};
    }

    MPI_Datatype mpi_pixel;
    MPI_Type_contiguous(4, MPI_FLOAT, &mpi_pixel);
    MPI_Type_commit(&mpi_pixel);
    MPI_Op depth_op;
    MPI_Op_create(depth_composite_op, 1, &depth_op);

    std::vector<Pixel> final_pixels;
    if (rank == 0) final_pixels.resize(pixel_count);

    MPI_Reduce(local_pixels.data(), rank == 0 ? final_pixels.data() : nullptr,
               int(pixel_count), mpi_pixel, depth_op, 0, MPI_COMM_WORLD);

    MPI_Op_free(&depth_op);
    MPI_Type_free(&mpi_pixel);

    if (rank == 0) {
      std::fprintf(stderr, "Writing %s...\n", output.c_str());
      std::ofstream out(output, std::ios::binary);
      out << "P6\n" << width << " " << height << "\n255\n";
      std::vector<unsigned char> row(size_t(width) * 3);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const Pixel &p = final_pixels[y * width + x];
          auto to_u8 = [exposure](float v) -> unsigned char {
            v = aces_filmic(v * exposure);
            v = std::pow(v, 1.f / 2.2f);
            return static_cast<unsigned char>(v * 255.f + 0.5f);
          };
          row[3*x] = to_u8(p.r);
          row[3*x+1] = to_u8(p.g);
          row[3*x+2] = to_u8(p.b);
        }
        out.write(reinterpret_cast<char*>(row.data()), std::streamsize(row.size()));
      }
      std::fprintf(stderr, "Output: %s\n", output.c_str());
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

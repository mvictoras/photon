#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "photon/pbrt/pbrt_parser.h"
#include "photon/pbrt/pbrt_to_photon.h"
#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/denoiser.h"
#include "photon/pt/pathtracer.h"

namespace {

// ACES filmic tone mapping (Narkowicz 2015 fit)
static float aces_filmic(float x)
{
  const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  x = std::fmax(x, 0.f);
  return std::fmin((x * (a * x + b)) / (x * (c * x + d) + e), 1.f);
}

struct Args {
  std::string scene_path;
  std::string output = "mpi_render.ppm";
  int width = -1;
  int height = -1;
  int spp = -1;
  int max_depth = -1;
  bool denoise = false;
  float exposure = 1.f;
};

Args parse_args(int argc, char **argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc)
      a.output = argv[++i];
    else if (arg == "--width" && i + 1 < argc)
      a.width = std::atoi(argv[++i]);
    else if (arg == "--height" && i + 1 < argc)
      a.height = std::atoi(argv[++i]);
    else if (arg == "--spp" && i + 1 < argc)
      a.spp = std::atoi(argv[++i]);
    else if (arg == "--max-depth" && i + 1 < argc)
      a.max_depth = std::atoi(argv[++i]);
    else if (arg == "--denoise")
      a.denoise = true;
    else if (arg == "--exposure" && i + 1 < argc)
      a.exposure = std::atof(argv[++i]);
    else if (arg[0] != '-')
      a.scene_path = arg;
  }
  return a;
}

struct Pixel {
  float r, g, b;
  float depth;
};

void depth_composite_op(void *invec, void *inoutvec, int *len, MPI_Datatype *)
{
  auto *in = static_cast<Pixel *>(invec);
  auto *out = static_cast<Pixel *>(inoutvec);
  for (int i = 0; i < *len; ++i) {
    if (in[i].depth > 0.f && (out[i].depth <= 0.f || in[i].depth < out[i].depth)) {
      out[i] = in[i];
    }
  }
}

void write_ppm(const std::string &file, const std::vector<Pixel> &pixels,
    uint32_t w, uint32_t h, float exposure)
{
  std::ofstream out(file, std::ios::binary);
  out << "P6\n" << w << " " << h << "\n255\n";

  std::vector<unsigned char> row(size_t(w) * 3);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const Pixel &p = pixels[y * w + x];

      auto to_u8 = [exposure](float v) -> unsigned char {
        v = aces_filmic(v * exposure);
        v = std::pow(v, 1.f / 2.2f);
        return static_cast<unsigned char>(v * 255.f + 0.5f);
      };

      row[3 * x + 0] = to_u8(p.r);
      row[3 * x + 1] = to_u8(p.g);
      row[3 * x + 2] = to_u8(p.b);
    }
    out.write(reinterpret_cast<const char *>(row.data()), std::streamsize(row.size()));
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
    Args args = parse_args(argc, argv);
    if (args.scene_path.empty()) {
      if (rank == 0)
        std::fprintf(stderr, "Usage: mpirun -np N photon_mpi_render <scene.pbrt> [options]\n");
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }

    if (rank == 0) {
      std::fprintf(stderr, "Data-parallel rendering with %d MPI rank(s)\n", num_ranks);
    }

    auto pbrt_scene = photon::pbrt::parse_pbrt_file(args.scene_path);

    if (args.width > 0) pbrt_scene.width = args.width;
    if (args.height > 0) pbrt_scene.height = args.height;
    if (args.spp > 0) pbrt_scene.spp = args.spp;
    if (args.max_depth > 0) pbrt_scene.max_depth = args.max_depth;

    const size_t total_meshes = pbrt_scene.meshes.size();
    const size_t meshes_per_rank = (total_meshes + num_ranks - 1) / num_ranks;
    const size_t my_start = std::min(size_t(rank) * meshes_per_rank, total_meshes);
    const size_t my_end = std::min(my_start + meshes_per_rank, total_meshes);

    photon::pbrt::PbrtScene local_scene = pbrt_scene;
    local_scene.meshes.clear();
    for (size_t i = my_start; i < my_end; ++i)
      local_scene.meshes.push_back(pbrt_scene.meshes[i]);

    if (rank == 0) {
      std::fprintf(stderr, "Rank %d: meshes [%zu-%zu) of %zu, %dx%d @ %d spp\n",
          rank, my_start, my_end, total_meshes,
          local_scene.width, local_scene.height, local_scene.spp);
    }

    std::string base_dir;
    auto last_slash = args.scene_path.find_last_of("/\\");
    if (last_slash != std::string::npos)
      base_dir = args.scene_path.substr(0, last_slash + 1);

    auto converted = photon::pbrt::convert_pbrt_scene(local_scene, base_dir);

    std::fprintf(stderr, "Rank %d: %u triangles, %u materials\n",
        rank, converted.scene.mesh.triangle_count(), converted.scene.material_count);

    auto backend = photon::pt::create_best_backend();
    backend->build_accel(converted.scene);

    photon::pt::PathTracer pt;
    pt.params.width = uint32_t(local_scene.width);
    pt.params.height = uint32_t(local_scene.height);
    pt.params.samples_per_pixel = uint32_t(local_scene.spp);
    pt.params.max_depth = uint32_t(local_scene.max_depth);
    pt.camera = converted.camera;
    pt.set_scene(converted.scene);
    pt.set_backend(std::move(backend));

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    auto result = pt.render();

    auto t1 = std::chrono::high_resolution_clock::now();
    double render_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::fprintf(stderr, "Rank %d: rendered in %.1f ms\n", rank, render_ms);

    const uint32_t w = pt.params.width;
    const uint32_t h = pt.params.height;
    const size_t pixel_count = size_t(w) * h;

    auto color_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);
    auto depth_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.depth);

    std::vector<Pixel> local_pixels(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i) {
      uint32_t y = uint32_t(i) / w;
      uint32_t x = uint32_t(i) % w;
      auto c = color_h(y, x);
      float d = depth_h(y, x);
      local_pixels[i] = {c.x, c.y, c.z, d};
    }

    MPI_Datatype mpi_pixel;
    MPI_Type_contiguous(4, MPI_FLOAT, &mpi_pixel);
    MPI_Type_commit(&mpi_pixel);

    MPI_Op depth_op;
    MPI_Op_create(depth_composite_op, 1, &depth_op);

    std::vector<Pixel> final_pixels;
    if (rank == 0)
      final_pixels.resize(pixel_count);

    auto t2 = std::chrono::high_resolution_clock::now();

    MPI_Reduce(local_pixels.data(), rank == 0 ? final_pixels.data() : nullptr,
               int(pixel_count), mpi_pixel, depth_op, 0, MPI_COMM_WORLD);

    auto t3 = std::chrono::high_resolution_clock::now();
    double composite_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    MPI_Op_free(&depth_op);
    MPI_Type_free(&mpi_pixel);

    if (rank == 0) {
      std::fprintf(stderr, "Composited in %.1f ms\n", composite_ms);
      write_ppm(args.output, final_pixels, w, h, args.exposure);
      std::fprintf(stderr, "Output: %s\n", args.output.c_str());
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

#include "photon/anari/PhotonDevice.h"
#include "photon/anari/SceneFromAnari.h"

#include <Kokkos_Core.hpp>

#include <anari/anari_cpp.hpp>
#include <anari/anari.h>

#include "photon/pt/scene.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using photon::pt::Vec3;

    photon::anari_device::PhotonDevice dev(nullptr);

    anari::Device d = (ANARIDevice)&dev;

    auto world = anari::newObject<anari::World>(d);

    const float positions[9] = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    const float colors[12] = {1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f};
    const uint32_t indices[3] = {0u, 1u, 2u};

    auto geom = anari::newObject<anari::Geometry>(d, "triangle");
    anari::setAndReleaseParameter(d,
        geom,
        "vertex.position",
        anari::newArray1D(d, positions, nullptr, nullptr, ANARI_FLOAT32_VEC3, 3));
    anari::setAndReleaseParameter(
        d, geom, "vertex.color", anari::newArray1D(d, colors, nullptr, nullptr, ANARI_FLOAT32_VEC4, 3));
    anari::setAndReleaseParameter(
        d, geom, "primitive.index", anari::newArray1D(d, indices, nullptr, nullptr, ANARI_UINT32_VEC3, 1));
    anari::commitParameters(d, geom);

    auto surface = anari::newObject<anari::Surface>(d);
    anari::setAndReleaseParameter(d, surface, "geometry", geom);

    auto mat = anari::newObject<anari::Material>(d, "matte");
    anari::setParameter(d, mat, "color", "color");
    anari::commitParameters(d, mat);
    anari::setAndReleaseParameter(d, surface, "material", mat);
    anari::commitParameters(d, surface);

    anari::setAndReleaseParameter(d, world, "surface", anari::newArray1D(d, &surface));
    anari::release(d, surface);
    anari::commitParameters(d, world);

    auto sceneOpt = photon::anari_device::build_scene_from_anari((ANARIWorld)world, dev);
    assert(sceneOpt.has_value());

    const auto &scene = *sceneOpt;
    assert(scene.material_count == 1);
    assert(scene.light_count == 0);

    assert(scene.mesh.positions.extent(0) == 3);
    assert(scene.mesh.indices.extent(0) == 3);
    assert(scene.mesh.material_ids.extent(0) == 1);

    assert(scene.mesh.albedo_per_prim.extent(0) == 1);
    auto albedoHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.albedo_per_prim);
    Vec3 a = albedoHost(0);
    assert(std::fabs(a.x - (1.f / 3.f)) < 1e-4f);
    assert(std::fabs(a.y - (1.f / 3.f)) < 1e-4f);
    assert(std::fabs(a.z - (1.f / 3.f)) < 1e-4f);

    anari::release(d, world);
  }
  Kokkos::finalize();
  return 0;
}

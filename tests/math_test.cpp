#include "photon/pt/math.h"
#include "photon/pt/math_mat4.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/color.h"

#include <Kokkos_Core.hpp>

#include <cassert>

namespace {

using photon::pt::f32;
using photon::pt::Mat4;
using photon::pt::Vec2;
using photon::pt::Vec3;

KOKKOS_FUNCTION inline bool near(f32 a, f32 b, f32 eps = 1e-5f)
{
  return Kokkos::fabs(a - b) <= eps;
}

KOKKOS_FUNCTION inline bool near(Vec3 a, Vec3 b, f32 eps = 1e-5f)
{
  return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps);
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    {
      Vec3 v{1.f, -2.f, 3.f};
      Vec3 n = -v;
      assert(n.x == -1.f && n.y == 2.f && n.z == -3.f);

      Vec3 d = v / 2.f;
      assert(d.x == 0.5f && d.y == -1.f && d.z == 1.5f);

      assert(photon::pt::max_component(v) == 3.f);
      assert(photon::pt::min_component(v) == -2.f);

      Vec3 a = photon::pt::abs(v);
      assert(a.x == 1.f && a.y == 2.f && a.z == 3.f);

      Vec3 I{1.f, -1.f, 0.f};
      Vec3 N{0.f, 1.f, 0.f};
      Vec3 R = photon::pt::reflect(I, N);
      assert(R.x == 1.f && R.y == 1.f && R.z == 0.f);

      Vec3 Ii{0.f, -1.f, 0.f};
      Vec3 T = photon::pt::refract(Ii, N, 1.f / 1.5f);
      assert(near(T, Vec3{0.f, -1.f, 0.f}));

      Vec3 I2 = photon::pt::normalize(Vec3{1.f, -0.01f, 0.f});
      Vec3 T2 = photon::pt::refract(I2, Vec3{0.f, 1.f, 0.f}, 1.5f);
      assert(T2.x == 0.f && T2.y == 0.f && T2.z == 0.f);

      Vec3 l = photon::pt::lerp(Vec3{0.f, 0.f, 0.f}, Vec3{10.f, 20.f, 30.f}, 0.5f);
      assert(l.x == 5.f && l.y == 10.f && l.z == 15.f);
    }

    {
      Vec2 a{1.f, 2.f};
      Vec2 b{3.f, -4.f};
      Vec2 c = a + b;
      assert(c.x == 4.f && c.y == -2.f);

      Vec2 d = a - b;
      assert(d.x == -2.f && d.y == 6.f);

      Vec2 e = a * 2.f;
      assert(e.x == 2.f && e.y == 4.f);

      Vec2 f = 2.f * a;
      assert(f.x == 2.f && f.y == 4.f);

      Vec2 g = a / 2.f;
      assert(g.x == 0.5f && g.y == 1.f);

      assert(photon::pt::dot(a, b) == (1.f * 3.f + 2.f * -4.f));
    }

    {
      Mat4 I = Mat4::identity();
      Vec3 p{1.f, 2.f, 3.f};
      assert(near(I.transform_point(p), p));

      Mat4 T = Mat4::translate(Vec3{5.f, -2.f, 1.f});
      Vec3 pt = T.transform_point(p);
      assert(near(pt, Vec3{6.f, 0.f, 4.f}));

      Mat4 S = Mat4::scale(Vec3{2.f, 3.f, 4.f});
      Vec3 ps = S.transform_point(p);
      assert(near(ps, Vec3{2.f, 6.f, 12.f}));

      Mat4 M = T * S;
      Vec3 pm = M.transform_point(p);
      assert(near(pm, Vec3{7.f, 4.f, 13.f}));

      Mat4 invM = M.inverse();
      Vec3 pr = invM.transform_point(pm);
      assert(near(pr, p));

      Vec3 d{1.f, 0.f, 0.f};
      Vec3 dt = T.transform_direction(d);
      assert(near(dt, d));

      Vec3 dn = S.transform_normal(Vec3{0.f, 1.f, 0.f});
      assert(near(dn, Vec3{0.f, 1.f / 3.f, 0.f}));
    }

    {
      Vec3 cLin{0.25f, 0.5f, 0.75f};
      Vec3 cSrgb = photon::pt::linear_to_srgb(cLin);
      Vec3 cBack = photon::pt::srgb_to_linear(cSrgb);
      assert(near(cBack, cLin, 2e-4f));

      f32 y = photon::pt::luminance(Vec3{1.f, 1.f, 1.f});
      assert(near(y, 1.f));

      Vec3 tone = photon::pt::aces_tonemap(Vec3{10.f, 10.f, 10.f});
      assert(tone.x <= 1.f && tone.y <= 1.f && tone.z <= 1.f);
      assert(tone.x >= 0.f && tone.y >= 0.f && tone.z >= 0.f);
    }
  }
  Kokkos::finalize();
  return 0;
}

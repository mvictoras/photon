#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

struct Mat4
{
  f32 m[4][4]{};

  static KOKKOS_FUNCTION Mat4 identity()
  {
    Mat4 r;
    r.m[0][0] = 1.f;
    r.m[1][1] = 1.f;
    r.m[2][2] = 1.f;
    r.m[3][3] = 1.f;
    return r;
  }

  static KOKKOS_FUNCTION Mat4 translate(const Vec3 &t)
  {
    Mat4 r = identity();
    r.m[0][3] = t.x;
    r.m[1][3] = t.y;
    r.m[2][3] = t.z;
    return r;
  }

  static KOKKOS_FUNCTION Mat4 scale(const Vec3 &s)
  {
    Mat4 r{};
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    r.m[3][3] = 1.f;
    return r;
  }

  static KOKKOS_FUNCTION Mat4 rotate(f32 angle_deg, const Vec3 &axis)
  {
    const f32 rad = angle_deg * 3.14159265358979323846f / 180.f;
    const f32 c = Kokkos::cos(rad);
    const f32 s = Kokkos::sin(rad);
    const Vec3 a = normalize(axis);
    const f32 t = 1.f - c;

    Mat4 r{};
    r.m[0][0] = t * a.x * a.x + c;
    r.m[0][1] = t * a.x * a.y - s * a.z;
    r.m[0][2] = t * a.x * a.z + s * a.y;
    r.m[1][0] = t * a.x * a.y + s * a.z;
    r.m[1][1] = t * a.y * a.y + c;
    r.m[1][2] = t * a.y * a.z - s * a.x;
    r.m[2][0] = t * a.x * a.z - s * a.y;
    r.m[2][1] = t * a.y * a.z + s * a.x;
    r.m[2][2] = t * a.z * a.z + c;
    r.m[3][3] = 1.f;
    return r;
  }

  static KOKKOS_FUNCTION Mat4 look_at(const Vec3 &eye, const Vec3 &target, const Vec3 &up)
  {
    const Vec3 fwd = normalize(target - eye);
    const Vec3 right = normalize(cross(fwd, up));
    const Vec3 new_up = cross(right, fwd);

    Mat4 r = identity();
    r.m[0][0] = right.x;   r.m[0][1] = new_up.x; r.m[0][2] = -fwd.x; r.m[0][3] = eye.x;
    r.m[1][0] = right.y;   r.m[1][1] = new_up.y; r.m[1][2] = -fwd.y; r.m[1][3] = eye.y;
    r.m[2][0] = right.z;   r.m[2][1] = new_up.z; r.m[2][2] = -fwd.z; r.m[2][3] = eye.z;
    return r;
  }

  KOKKOS_FUNCTION Vec3 transform_point(const Vec3 &p) const
  {
    const f32 x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
    const f32 y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
    const f32 z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
    const f32 w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];

    if (w != 0.f && w != 1.f) {
      const f32 invw = 1.f / w;
      return {x * invw, y * invw, z * invw};
    }

    return {x, y, z};
  }

  KOKKOS_FUNCTION Vec3 transform_direction(const Vec3 &d) const
  {
    const f32 x = m[0][0] * d.x + m[0][1] * d.y + m[0][2] * d.z;
    const f32 y = m[1][0] * d.x + m[1][1] * d.y + m[1][2] * d.z;
    const f32 z = m[2][0] * d.x + m[2][1] * d.y + m[2][2] * d.z;
    return {x, y, z};
  }

  KOKKOS_FUNCTION Vec3 transform_normal(const Vec3 &n) const
  {
    Mat4 inv = inverse();
    const f32 x = inv.m[0][0] * n.x + inv.m[1][0] * n.y + inv.m[2][0] * n.z;
    const f32 y = inv.m[0][1] * n.x + inv.m[1][1] * n.y + inv.m[2][1] * n.z;
    const f32 z = inv.m[0][2] * n.x + inv.m[1][2] * n.y + inv.m[2][2] * n.z;
    return {x, y, z};
  }

  KOKKOS_FUNCTION Mat4 inverse() const
  {
    Mat4 r = identity();

    const f32 a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    const f32 a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    const f32 a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

    const f32 det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);

    if (det == 0.f)
      return {};

    const f32 invdet = 1.f / det;

    r.m[0][0] = (a11 * a22 - a12 * a21) * invdet;
    r.m[0][1] = (a02 * a21 - a01 * a22) * invdet;
    r.m[0][2] = (a01 * a12 - a02 * a11) * invdet;

    r.m[1][0] = (a12 * a20 - a10 * a22) * invdet;
    r.m[1][1] = (a00 * a22 - a02 * a20) * invdet;
    r.m[1][2] = (a02 * a10 - a00 * a12) * invdet;

    r.m[2][0] = (a10 * a21 - a11 * a20) * invdet;
    r.m[2][1] = (a01 * a20 - a00 * a21) * invdet;
    r.m[2][2] = (a00 * a11 - a01 * a10) * invdet;

    const f32 tx = m[0][3];
    const f32 ty = m[1][3];
    const f32 tz = m[2][3];

    r.m[0][3] = -(r.m[0][0] * tx + r.m[0][1] * ty + r.m[0][2] * tz);
    r.m[1][3] = -(r.m[1][0] * tx + r.m[1][1] * ty + r.m[1][2] * tz);
    r.m[2][3] = -(r.m[2][0] * tx + r.m[2][1] * ty + r.m[2][2] * tz);

    return r;
  }

  KOKKOS_FUNCTION Mat4 operator*(const Mat4 &b) const
  {
    Mat4 r{};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        f32 s = 0.f;
        for (int k = 0; k < 4; ++k)
          s += m[i][k] * b.m[k][j];
        r.m[i][j] = s;
      }
    }
    return r;
  }
};

}

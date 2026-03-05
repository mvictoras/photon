#pragma once

#include <Kokkos_Core.hpp>

namespace photon::pt {

using f32 = float;
using u32 = unsigned;

struct Vec3
{
  f32 x{0.f};
  f32 y{0.f};
  f32 z{0.f};

  KOKKOS_FUNCTION Vec3() = default;
  KOKKOS_FUNCTION Vec3(f32 xx, f32 yy, f32 zz) : x(xx), y(yy), z(zz) {}

  KOKKOS_FUNCTION Vec3 &operator+=(const Vec3 &b)
  {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
};

KOKKOS_FUNCTION inline Vec3 operator+(Vec3 a, const Vec3 &b)
{
  a += b;
  return a;
}

KOKKOS_FUNCTION inline Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

KOKKOS_FUNCTION inline Vec3 operator-(const Vec3 &v)
{
  return {-v.x, -v.y, -v.z};
}

KOKKOS_FUNCTION inline Vec3 operator/(const Vec3 &a, f32 s)
{
  const f32 inv = 1.f / s;
  return {a.x * inv, a.y * inv, a.z * inv};
}

KOKKOS_FUNCTION inline Vec3 operator*(const Vec3 &a, f32 s)
{
  return {a.x * s, a.y * s, a.z * s};
}

KOKKOS_FUNCTION inline Vec3 operator*(f32 s, const Vec3 &a)
{
  return a * s;
}

KOKKOS_FUNCTION inline Vec3 operator*(const Vec3 &a, const Vec3 &b)
{
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}

KOKKOS_FUNCTION inline f32 dot(const Vec3 &a, const Vec3 &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

KOKKOS_FUNCTION inline Vec3 cross(const Vec3 &a, const Vec3 &b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

KOKKOS_FUNCTION inline f32 length(const Vec3 &v)
{
  return Kokkos::sqrt(dot(v, v));
}

KOKKOS_FUNCTION inline Vec3 normalize(const Vec3 &v)
{
  const f32 len = length(v);
  return len > 0.f ? (v * (1.f / len)) : Vec3{};
}

KOKKOS_FUNCTION inline Vec3 clamp01(const Vec3 &c)
{
  auto clamp = [](f32 x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); };
  return {clamp(c.x), clamp(c.y), clamp(c.z)};
}

KOKKOS_FUNCTION inline f32 max_component(const Vec3 &v)
{
  return v.x > v.y ? (v.x > v.z ? v.x : v.z) : (v.y > v.z ? v.y : v.z);
}

KOKKOS_FUNCTION inline f32 min_component(const Vec3 &v)
{
  return v.x < v.y ? (v.x < v.z ? v.x : v.z) : (v.y < v.z ? v.y : v.z);
}

KOKKOS_FUNCTION inline Vec3 reflect(const Vec3 &I, const Vec3 &N)
{
  return I - (2.f * dot(I, N)) * N;
}

KOKKOS_FUNCTION inline Vec3 refract(const Vec3 &I, const Vec3 &N, f32 eta)
{
  const f32 cosi = -dot(I, N);
  const f32 k = 1.f - eta * eta * (1.f - cosi * cosi);
  if (k < 0.f)
    return {};
  return eta * I + (eta * cosi - Kokkos::sqrt(k)) * N;
}

KOKKOS_FUNCTION inline Vec3 lerp(const Vec3 &a, const Vec3 &b, f32 t)
{
  return a + (b - a) * t;
}

KOKKOS_FUNCTION inline Vec3 abs(const Vec3 &v)
{
  return {Kokkos::fabs(v.x), Kokkos::fabs(v.y), Kokkos::fabs(v.z)};
}

} // namespace photon::pt

#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

struct Vec2
{
  f32 x{0.f};
  f32 y{0.f};

  KOKKOS_FUNCTION Vec2() = default;
  KOKKOS_FUNCTION Vec2(f32 xx, f32 yy) : x(xx), y(yy) {}
};

KOKKOS_FUNCTION inline Vec2 operator+(Vec2 a, const Vec2 &b)
{
  return {a.x + b.x, a.y + b.y};
}

KOKKOS_FUNCTION inline Vec2 operator-(const Vec2 &a, const Vec2 &b)
{
  return {a.x - b.x, a.y - b.y};
}

KOKKOS_FUNCTION inline Vec2 operator*(const Vec2 &a, f32 s)
{
  return {a.x * s, a.y * s};
}

KOKKOS_FUNCTION inline Vec2 operator*(f32 s, const Vec2 &a)
{
  return a * s;
}

KOKKOS_FUNCTION inline Vec2 operator/(const Vec2 &a, f32 s)
{
  const f32 inv = 1.f / s;
  return {a.x * inv, a.y * inv};
}

KOKKOS_FUNCTION inline f32 dot(const Vec2 &a, const Vec2 &b)
{
  return a.x * b.x + a.y * b.y;
}

}

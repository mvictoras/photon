#include "photon/pt/environment_map.h"

#include <algorithm>
#include <vector>

namespace photon::pt {

void EnvironmentMap::build_cdf()
{
  if (width == 0 || height == 0)
    return;

  auto hostPix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pixels);

  auto hostMarginal = Kokkos::create_mirror_view(marginal_cdf);
  auto hostConditional = Kokkos::create_mirror_view(conditional_cdf);

  std::vector<f32> rowWeights(height, 0.f);

  hostMarginal(0) = 0.f;
  for (u32 y = 0; y < height; ++y) {
    hostConditional(y, 0) = 0.f;

    const f32 theta = ((f32)y + 0.5f) * (PI / (f32)height);
    const f32 sin_theta = Kokkos::sin(theta);

    f32 rowSum = 0.f;
    for (u32 x = 0; x < width; ++x) {
      const f32 w = luminance(hostPix(y, x));
      rowSum += w;
      hostConditional(y, x + 1) = rowSum;
    }

    if (rowSum > 0.f) {
      const f32 invRow = 1.f / rowSum;
      for (u32 x = 0; x <= width; ++x)
        hostConditional(y, x) *= invRow;
      hostConditional(y, width) = 1.f;
    } else {
      for (u32 x = 0; x <= width; ++x)
        hostConditional(y, x) = (f32)x / (f32)width;
    }

    rowWeights[y] = rowSum * sin_theta;
  }

  f32 total = 0.f;
  for (u32 y = 0; y < height; ++y) {
    total += rowWeights[y];
    hostMarginal(y + 1) = total;
  }

  if (total > 0.f) {
    const f32 invTotal = 1.f / total;
    for (u32 y = 0; y <= height; ++y)
      hostMarginal(y) *= invTotal;
    hostMarginal(height) = 1.f;
  } else {
    for (u32 y = 0; y <= height; ++y)
      hostMarginal(y) = (f32)y / (f32)height;
  }

  Kokkos::deep_copy(marginal_cdf, hostMarginal);
  Kokkos::deep_copy(conditional_cdf, hostConditional);
}

}

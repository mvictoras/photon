#pragma once

#include "photon/pt/math.h"
#include "photon/pt/pathtracer.h"

#include <string>

namespace photon::pt::io {

// Core PPM (P6) writer: rgb points to width*height interleaved RGB floats
// (row-major). Pixels are tone mapped with aces_tonemap() and gamma_correct()
// from tone_mapping.h — the one ACES implementation in the codebase.
void write_ppm(const std::string &path, const float *rgb, u32 width, u32 height,
               f32 exposure = 1.f);

// Convenience overload for a PathTracer RenderResult: mirrors the color AOV
// to the host and writes it.
void write_ppm(const std::string &path, const RenderResult &result, u32 width,
               u32 height, f32 exposure = 1.f);

} // namespace photon::pt::io

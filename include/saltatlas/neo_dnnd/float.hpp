// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <limits>
#include <cmath>

namespace saltatlas::neo_dnnd::ndndtl {
template <typename T>
inline bool nearly_equal(const T a, const T b,
                         const double eps = std::numeric_limits<T>::epsilon()) {
  return (std::fabs(a - b) < eps);
}
}  // namespace saltatlas::neo_dnnd::utl
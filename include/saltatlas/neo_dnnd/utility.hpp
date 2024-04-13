// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <limits>
#include <utility>

#include <metall/detail/utilities.hpp>

namespace saltatlas::neo_dnnd::utility {

using metall::mtlldetail::partial_range;
using metall::mtlldetail::round_up;
using metall::mtlldetail::round_down;
using metall::mtlldetail::log_cpt;
using metall::mtlldetail::log2_dynamic;

/// \brief Compute min, max. mean, and variance.
template <typename T>
inline std::tuple<T, T, double, double> get_stats(const std::vector<T>& v) {
  assert(!v.empty());

  T min = std::numeric_limits<T>::max();
  T max = std::numeric_limits<T>::lowest();
  T sum = 0.0;
  T sum_sq = 0.0;

  for (const auto& e : v) {
    min = std::min(min, e);
    max = std::max(max, e);
    sum += e;
    sum_sq += e * e;
  }

  const double mean = double(sum) / v.size();
  const double var = double(sum_sq) / v.size() - mean * mean;

  return {min, max, mean, std::sqrt(var)};
}

}  // namespace saltatlas::neo_dnnd::dn2utl

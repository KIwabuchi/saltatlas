// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

#include <metall/detail/utilities.hpp>

namespace saltatlas::neo_dnnd::utility {

using metall::mtlldetail::log2_dynamic;
using metall::mtlldetail::log_cpt;
using metall::mtlldetail::partial_range;
using metall::mtlldetail::round_down;
using metall::mtlldetail::round_up;

/// \brief Compute min, max. mean, and standard deviation.
template <typename T>
inline std::tuple<T, T, double, double> get_stats(const std::vector<T>& v) {
  assert(!v.empty());

  T min    = std::numeric_limits<T>::max();
  T max    = std::numeric_limits<T>::lowest();
  T sum    = 0.0;
  T sum_sq = 0.0;

  for (const auto& e : v) {
    min = std::min(min, e);
    max = std::max(max, e);
    sum += e;
    sum_sq += e * e;
  }

  const double mean    = double(sum) / v.size();
  const double var     = double(sum_sq) / v.size() - mean * mean;
  const double std_var = std::sqrt(var);

  return {min, max, mean, std_var};
}

/// \brief Generate a round-robin tournament schedule.
/// \param num_players The number of players. Must be even.
/// \param payler_id The player ID. Must be in the range of [0, num_players).
/// \param out_itr An output iterator to store the opponent player IDs.
template <typename OutIterator>
constexpr void gen_round_robin_tournament(const std::size_t num_players,
                                          const std::size_t payler_id,
                                          OutIterator       out_itr) {
  if (num_players <= 1 || num_players % 2 != 0) {
    return;
  }

  // Generate pairs using the round-robin tournament algorithm.
  for (std::size_t round = 0; round < num_players - 1; ++round) {
    for (std::size_t i = 0; i < num_players / 2; ++i) {
      const std::size_t pair1 =
          (i == 0) ? 0 : ((round + i) % (num_players - 1)) + 1;
      const std::size_t pair2 =
          (round - i + num_players - 1) % (num_players - 1) + 1;

      // If one of the players is me, the other player is my opponent in this
      // round.
      if (pair1 == payler_id) {
        *out_itr = pair2;
        ++out_itr;
      } else if (pair2 == payler_id) {
        *out_itr = pair1;
        ++out_itr;
      }
    }
  }
}

}  // namespace saltatlas::neo_dnnd::utility

// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <limits>
#include <utility>

#include "saltatlas/neo_dnnd/builtin_function.hpp"

#ifndef DNND2_PRAGMA_IGNORE_GCC_UNINIT_WARNING_BEGIN
#if defined(__GNUC__) and !defined(__clang__)
#define DNND2_PRAGMA_IGNORE_GCC_UNINIT_WARNING_BEGIN \
  _Pragma("GCC diagnostic push")                      \
      _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#else
#define DNND2_PRAGMA_IGNORE_GCC_UNINIT_WARNING_BEGIN
#endif
#endif

#ifndef DNND2_PRAGMA_IGNORE_GCC_UNINIT_WARNING_END
#if defined(__GNUC__) and !defined(__clang__)
#define DNND2_PRAGMA_IGNORE_GCC_UNINIT_WARNING_END \
  _Pragma("GCC diagnostic pop")
#else
#define DNND2_PRAGMA_IGNORE_GCC_UNINIT_WARNING_END
#endif
#endif

namespace saltatlas::neo_dnnd::utility {

/// \brief Divides a length into multiple groups.
/// \param length A length to be divided.
/// \param block_no A block number.
/// \param num_blocks The number of total blocks.
/// \return The begin and end index of the range. Note that [begin, end).
inline std::pair<std::size_t, std::size_t> partial_range(
    const std::size_t length, const std::size_t block_no,
    const std::size_t num_blocks) {
  std::size_t partial_length = length / num_blocks;
  std::size_t r = length % num_blocks;

  std::size_t begin_index;

  if (block_no < r) {
    begin_index = (partial_length + 1) * block_no;
    ++partial_length;
  } else {
    begin_index = (partial_length + 1) * r + partial_length * (block_no - r);
  }

  return std::make_pair(begin_index, begin_index + partial_length);
}

/// \brief Rounds up to the nearest multiple of 'base'.
/// base must not be 0.
/// \param to_round
/// \param base
/// \return
inline constexpr int64_t round_up(const int64_t to_round,
                                  const int64_t base) noexcept {
  return ((to_round + static_cast<int64_t>(to_round >= 0) * (base - 1)) /
          base) *
         base;
}

/// \brief Rounds down to the nearest multiple of 'base'.
/// base must not be 0.
/// \param to_round
/// \param base
/// \return
inline constexpr int64_t round_down(const int64_t to_round,
                                    const int64_t base) noexcept {
  return ((to_round - static_cast<int64_t>(to_round < 0) * (base - 1)) / base) *
         base;
}

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

/// \brief Computes the base 'base' logarithm of 'n' at compile time.
/// NOTE that this method causes a recursive function call if non-constexpr
/// value is given \param n Input value \param base Base \return Returns the
/// base 'base' logarithm of 'n'
inline constexpr uint64_t log_cpt(const uint64_t n,
                                  const uint64_t base) noexcept {
  return (n < base) ? 0 : 1 + log_cpt(n / base, base);
}

/// \brief Compute log2 using a builtin function in order to avoid
/// loop/recursive operation happen in utility::log. If 'n' is 0 or not a power
/// of 2, the result is invalid. \param n Input value \return Returns base 2
/// logarithm of 2
inline uint64_t log2_dynamic(const uint64_t n) noexcept {
  assert(n && !(n & (n - 1)));  // n > 0 && a power of 2
  return saltatlas::neo_dnnd::ndndtl::ctzll(n);
}

}  // namespace saltatlas::neo_dnnd::dn2utl

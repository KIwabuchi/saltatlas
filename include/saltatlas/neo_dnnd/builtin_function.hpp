// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

namespace saltatlas::neo_dnnd::ndndtl {
/// \brief Count Leading Zeros.
inline int clzll(const unsigned long long x) noexcept {
#if defined(__GNUG__) || defined(__clang__)
  return __builtin_clzll(x);
#else
#error "GCC or Clang must be used to use __builtin_clzll" << std::endl;
#endif
}

/// \brief Count Trailing Zeros.
inline int ctzll(const unsigned long long x) noexcept {
#if defined(__GNUG__) || defined(__clang__)
  return __builtin_ctzll(x);
#else
#error "GCC or Clang must be used to use __builtin_ctzll" << std::endl;
#endif
}
}
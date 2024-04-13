// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

#include "saltatlas/neo_dnnd/blas.hpp"
#include "saltatlas/neo_dnnd/float.hpp"

namespace saltatlas::neo_dnnd::distance {
template <typename feature_element_type, typename distance_type>
using similarity_type = distance_type(const feature_element_type *const,
                                      const std::size_t,
                                      const feature_element_type *const,
                                      const std::size_t);

template <typename feature_element_type, typename distance_type>
inline distance_type invalid(const feature_element_type *const,
                             const std::size_t,
                             const feature_element_type *const,
                             const std::size_t) {
  std::cerr << "Invalid distance function" << std::endl;
  std::abort();
  return distance_type{};
}

template <typename feature_element_type, typename distance_type>
inline distance_type l1(const feature_element_type *const f0,
                        const std::size_t                 len0,
                        const feature_element_type *const f1,
                        const std::size_t                 len1) {
  assert(len0 == len1);
  distance_type d = 0;
  for (std::size_t i = 0; i < len0; ++i) {
    const auto x = std::abs(f0[i] - f1[i]);
    d += x;
  }
  return static_cast<distance_type>(std::sqrt(d));
}

template <typename feature_element_type, typename distance_type>
inline distance_type l2(const feature_element_type *const f0,
                        const std::size_t                 len0,
                        const feature_element_type *const f1,
                        const std::size_t                 len1) {
  assert(len0 == len1);
  distance_type d = 0;
  for (std::size_t i = 0; i < len0; ++i) {
    const auto x = (f0[i] - f1[i]);
    d += x * x;
  }
  return static_cast<distance_type>(std::sqrt(d));
}

/// \brief Squared Euclidean distance, which omits the final square root in the
/// calculation of l2 norm.
template <typename feature_element_type, typename distance_type>
inline distance_type sql2(const feature_element_type *const f0,
                          const std::size_t                 len0,
                          const feature_element_type *const f1,
                          const std::size_t                 len1) {
  assert(len0 == len1);
  distance_type d = 0;
  for (std::size_t i = 0; i < len0; ++i) {
    const auto x = (f0[i] - f1[i]);
    d += x * x;
  }
  return d;
}

template <typename feature_element_type, typename distance_type>
inline distance_type cosine(const feature_element_type *const f0,
                            const std::size_t                 len0,
                            const feature_element_type *const f1,
                            const std::size_t                 len1) {
  assert(len0 == len1);
  const distance_type n0 = ndndtl::blas::inner_product(len0, f0, f0);
  const distance_type n1 = ndndtl::blas::inner_product(len1, f1, f1);
  if (ndndtl::nearly_equal(n0, distance_type(0)) &&
      ndndtl::nearly_equal(n1, distance_type(0)))
    return static_cast<distance_type>(0);
  else if (ndndtl::nearly_equal(n0, distance_type(0)) ||
           ndndtl::nearly_equal(n1, distance_type(0)))
    return static_cast<distance_type>(1);

  const distance_type x = ndndtl::blas::inner_product(len0, f0, f1);
  return static_cast<distance_type>(1.0 - x / std::sqrt(n0 * n1));
}

/// \brief Alternative cosine distance. The original model is from PyNNDescent.
/// This function returns the same relative distance orders as the normal
/// cosine similarity.
template <typename feature_element_type, typename distance_type>
inline distance_type alt_cosine(const feature_element_type *const f0,
                                const std::size_t                 len0,
                                const feature_element_type *const f1,
                                const std::size_t                 len1) {
  assert(len0 == len1);
  const distance_type n0 = ndndtl::blas::inner_product(len0, f0, f0);
  const distance_type n1 = ndndtl::blas::inner_product(len1, f1, f1);
  if (ndndtl::nearly_equal(n0, distance_type(0)) &&
      ndndtl::nearly_equal(n1, distance_type(0))) {
    return static_cast<distance_type>(0);
  } else if (ndndtl::nearly_equal(n0, distance_type(0)) ||
             ndndtl::nearly_equal(n1, distance_type(0))) {
    // Does not return the max value to prevent overflow on the caller side.
    return std::numeric_limits<distance_type>::max() / distance_type(2);
  }

  const distance_type x = ndndtl::blas::inner_product(len0, f0, f1);
  if (x < 0 || ndndtl::nearly_equal(x, distance_type(0))) {
    return std::numeric_limits<distance_type>::max() / distance_type(2);
  }

  return static_cast<distance_type>(std::log2(std::sqrt(n0 * n1) / x));
}

template <typename feature_element_type, typename distance_type>
inline distance_type jaccard_index(const feature_element_type *const f0,
                                   const std::size_t                 len0,
                                   const feature_element_type *const f1,
                                   const std::size_t                 len1) {
  assert(len0 == len1);
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < len0; ++i) {
    const bool x_true = !!f0[i];
    const bool y_true = !!f1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return distance_type{0};
  else
    return static_cast<distance_type>(num_non_zero - num_equal) /
           static_cast<distance_type>(num_non_zero);
}

/// \brief Alternative Jaccard index. The original model is from PyNNDescent.
/// This function returns the same relative distance orders as the normal
/// Jaccard index.
template <typename feature_element_type, typename distance_type>
inline distance_type alt_jaccard_index(const feature_element_type *const f0,
                                       const std::size_t                 len0,
                                       const feature_element_type *const f1,
                                       const std::size_t                 len1) {
  assert(len0 == len1);
  std::size_t num_non_zero = 0;
  std::size_t num_equal    = 0;
  for (std::size_t i = 0; i < len0; ++i) {
    const bool x_true = !!f0[i];
    const bool y_true = !!f1[i];
    num_non_zero += x_true | y_true;
    num_equal += x_true & y_true;
  }

  if (num_non_zero == 0)
    return distance_type{0};
  else
    return static_cast<distance_type>(
        -std::log2(distance_type(num_equal) / distance_type(num_non_zero)));
}

template <typename feature_element_type, typename distance_type>
inline distance_type levenshtein(const feature_element_type *const s0,
                                 const std::size_t                 m,
                                 const feature_element_type *const s1,
                                 const std::size_t                 n) {
  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }

  // Row of matrix for dynamic programming approach
  std::vector<size_t> dist_row(m + 1);
  for (size_t i = 0; i < m + 1; ++i) {
    dist_row[i] = i;
  }

  for (size_t i = 1; i < n + 1; ++i) {
    size_t diag = i - 1;
    size_t next_diag;
    dist_row[0] = i;
    for (size_t j = 1; j < m + 1; ++j) {
      next_diag              = dist_row[j];
      bool substitution_cost = (s0[j - 1] != s1[i - 1]);

      dist_row[j] =
          std::min(1 + dist_row[j],
                   std::min(1 + dist_row[j - 1], substitution_cost + diag));
      diag = next_diag;
    }
  }

  return static_cast<distance_type>(dist_row[m]);
}

enum class similarity_id : uint8_t {
  invalid,
  l1,
  l2,
  sql2,
  cosine,
  altcosine,
  jaccard,
  altjaccard,
  levenshtein
};

inline similarity_id convert_to_similarity_id(
    const std::string_view &similarity_name) {
  if (similarity_name == "l1") {
    return similarity_id::l1;
  } else if (similarity_name == "l2") {
    return similarity_id::l2;
  } else if (similarity_name == "sql2") {
    return similarity_id::sql2;
  } else if (similarity_name == "cosine") {
    return similarity_id::cosine;
  } else if (similarity_name == "altcosine") {
    return similarity_id::altcosine;
  } else if (similarity_name == "jaccard") {
    return similarity_id::jaccard;
  } else if (similarity_name == "altjaccard") {
    return similarity_id::altjaccard;
  } else if (similarity_name == "levenshtein") {
    return similarity_id::levenshtein;
  }
  return similarity_id::invalid;
}

inline std::string convert_to_similarity_name(const similarity_id &id) {
  switch (id) {
    case similarity_id::l1:
      return "l1";
    case similarity_id::l2:
      return "l2";
    case similarity_id::sql2:
      return "sql2";
    case similarity_id::cosine:
      return "cosine";
    case similarity_id::altcosine:
      return "altcosine";
    case similarity_id::jaccard:
      return "jaccard";
    case similarity_id::altjaccard:
      return "altjaccard";
    case similarity_id::levenshtein:
      return "levenshtein";
    default:
      return "invalid";
  }
}

template <typename feature_element_type, typename distance_type>
inline similarity_type<feature_element_type, distance_type> &
similarity_function(const similarity_id &id) {
  if (id == similarity_id::l1) {
    return l1<feature_element_type, distance_type>;
  } else if (id == similarity_id::l2) {
    return l2<feature_element_type, distance_type>;
  } else if (id == similarity_id::sql2) {
    return sql2<feature_element_type, distance_type>;
  } else if (id == similarity_id::cosine) {
    return cosine<feature_element_type, distance_type>;
  } else if (id == similarity_id::altcosine) {
    return alt_cosine<feature_element_type, distance_type>;
  } else if (id == similarity_id::jaccard) {
    return jaccard_index<feature_element_type, distance_type>;
  } else if (id == similarity_id::altjaccard) {
    return alt_jaccard_index<feature_element_type, distance_type>;
  } else if (id == similarity_id::levenshtein) {
    return levenshtein<feature_element_type, distance_type>;
  }
  return invalid<feature_element_type, distance_type>;
}

template <typename feature_element_type, typename distance_type>
inline similarity_type<feature_element_type, distance_type> &
similarity_function(const std::string_view similarity_name) {
  return similarity_function<feature_element_type, distance_type>(
      convert_to_similarity_id(similarity_name));
}
}  // namespace saltatlas::neo_dnnd::distance

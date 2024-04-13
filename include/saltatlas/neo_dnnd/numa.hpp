// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __APPLE__
#include <numa.h>
#define USE_NUMA_LIB
#else
#warning "Does not use NUMA"
#endif

namespace saltatlas::neo_dnnd::numa {

bool available() noexcept {
#ifdef USE_NUMA_LIB
  return ::numa_available() != -1;
#else
  return false;
#endif
}

int get_num_avail_nodes() noexcept {
#ifdef USE_NUMA_LIB
  return ::numa_max_node() + 1;
#else
  return 1;
#endif
}

}  // namespace saltatlas::neo_dnnd::numa
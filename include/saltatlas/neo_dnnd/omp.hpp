// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <metall/utility/open_mp.hpp>

namespace saltatlas::neo_dnnd {
namespace omp {

using metall::utility::omp::get_num_threads;
using metall::utility::omp::get_thread_num;
using metall::utility::omp::set_num_threads;

}  // namespace omp
}  // namespace saltatlas::neo_dnnd

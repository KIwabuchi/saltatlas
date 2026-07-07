// Copyright 2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#include "saltatlas/solanet/detail/apu_nn/graph_reverser.hpp"
#include "saltatlas/solanet/detail/apu_nn/nndescent.hpp"
#include "saltatlas/solanet/detail/apu_nn/search_knng_builder.hpp"
#include "saltatlas/solanet/solanet.hpp"

namespace {
using id_type   = std::uint64_t;
using fe_type   = float;
using dist_type = float;
using engine_type =
    saltatlas::solanet::solanet_engine_apu<id_type, fe_type, dist_type>;

static_assert(std::is_same_v<typename engine_type::id_type, id_type>);
static_assert(
    std::is_same_v<typename engine_type::neighbor_type::id_type, id_type>);
}  // namespace

template std::pair<saltatlas::solanet::apu_nn::matrix<std::uint64_t>,
                   saltatlas::solanet::apu_nn::matrix<float>>
saltatlas::solanet::apu_nn::build_index<std::uint64_t, float, float>(
    const saltatlas::solanet::apu_nn::matrix_view<float>&,
    const std::string_view, const int, const float, const float,
    const std::uint64_t, const int);

template saltatlas::solanet::apu_nn::csr_graph<std::uint64_t, float>
saltatlas::solanet::apu_nn::make_reversed_graph_apu<std::uint64_t, float>(
    const saltatlas::solanet::apu_nn::matrix_view<std::uint64_t>&,
    const saltatlas::solanet::apu_nn::matrix_view<float>&, const std::size_t);

template void saltatlas::solanet::apu_nn::make_optimized_query_graph_apu<
    std::uint64_t, float>(
    const saltatlas::solanet::apu_nn::matrix_view<std::uint64_t>&,
    const saltatlas::solanet::apu_nn::matrix_view<float>&,
    saltatlas::solanet::apu_nn::matrix_view<std::uint64_t>,
    const saltatlas::solanet::apu_nn::query_graph_distance_mode);

int main() { return 0; }

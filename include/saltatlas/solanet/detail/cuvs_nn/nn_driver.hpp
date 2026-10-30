// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#define SALTATLAS_SOLANET_APU_NND_OPTIMIZE_QUERY_GRAPH

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include "saltatlas/dnnd/utility.hpp"
#include "saltatlas/solanet/detail/nn_index_view.hpp"
#include "saltatlas/solanet/detail/point_store_view.hpp"
#include "saltatlas/solanet/detail/utilities/raft_utils.hpp"
#if defined(SALTATLAS_SOLANET_CUVS)
#include "saltatlas/solanet/detail/cuvs_nn/memory.hpp"
#include "saltatlas/solanet/detail/cuvs_nn/utils.hpp"
#elif defined(SALTATLAS_SOLANET_APU)
#include "saltatlas/solanet/detail/apu_nn/memory.hpp"
#include "saltatlas/solanet/detail/apu_nn/nndescent.hpp"
#include "saltatlas/solanet/detail/apu_nn/search_knng_builder.hpp"
#endif

namespace saltatlas::solanet::cuvs_nn {

namespace slntrft = saltatlas::solanet::raft_utils;

template <typename IDType, typename FeatureElemType, typename DistanceType>
class driver {
 public:
  using id_type   = IDType;
  using dist_type = DistanceType;
  using fe_type   = FeatureElemType;

  using point_store_view_type = d3dtl::point_store_view<id_type, fe_type>;
  using point_type            = typename point_store_view_type::point_type;
  using const_point_type = typename point_store_view_type::const_point_type;
  using point_store_type = raft_utils::h_matrix_type<fe_type>;
  using index_view_type  = d3dtl::nn_index_view<IDType, DistanceType>;
  using index_nid_type   = raft_utils::d_matrix_type<id_type>;
  using index_dist_type  = raft_utils::d_matrix_type<dist_type>;
  // Host-resident counterparts, e.g. for the final host-side conversion of
  // the (device-resident) master index into the output kNNG.
  using host_index_nid_type  = raft_utils::h_matrix_type<id_type>;
  using host_index_dist_type = raft_utils::h_matrix_type<dist_type>;
  using group_index_nid_type = host_index_nid_type;

  struct index_type {
    index_nid_type  nids;
    index_dist_type dists;
  };

  struct host_index_type {
    host_index_nid_type  nids;
    host_index_dist_type dists;
  };

 private:
  // CAGRA Index: dataset and knng
  using cagra_index_t = cuvs::neighbors::cagra::index<fe_type, id_type>;

 public:
  explicit driver(const bool verbose = false) : m_verbose(verbose) {}

  // Disable for now..
  driver(const driver&)             = delete;
  driver& operator=(const driver&)  = delete;
  driver(const driver&&)            = delete;
  driver& operator=(const driver&&) = delete;

  ~driver() { reset(); }

  // Allocate memory for receiving the point store and index data from other
  // ranks
  auto alloc_recv_buf(const size_t size_bytes) {
#if defined(SALTATLAS_SOLANET_CUVS)
    return make_cuda_host_array<unsigned char>(size_bytes);
#else
    return make_hip_array<unsigned char>(size_bytes);
#endif
  }

  host_index_type copy_index_to_host(const index_type& index) {
    host_index_type host_index;
    host_index.nids =
        raft_utils::copy_to_host(index.nids.view(), m_host_res, m_dev_res);
    host_index.dists =
        raft_utils::copy_to_host(index.dists.view(), m_host_res, m_dev_res);
    return host_index;
  }

  // MPI windows must expose host memory. Refresh the fixed window backing
  // storage after each device-side index update.
  void copy_index_nids_to_host(const index_type& index, id_type* host_nids) {
    const auto bytes = index.nids.size() * sizeof(id_type);
    if (bytes != 0) {
      SOLANET_CUDA_CHECK(cudaDeviceSynchronize());
      // host_nids backs an active MPI window after the first merge level.
      // Keep CUDA transfers off that memory while MPI owns its registration.
      constexpr size_t staging_bytes = 16ULL << 20;
      auto             staging =
          make_cuda_host_array<unsigned char>(std::min(bytes, staging_bytes));
      const auto* device_nids =
          reinterpret_cast<const unsigned char*>(index.nids.data_handle());
      auto* destination = reinterpret_cast<unsigned char*>(host_nids);
      for (size_t offset = 0; offset < bytes; offset += staging_bytes) {
        const auto count = std::min(staging_bytes, bytes - offset);
        SOLANET_CUDA_CHECK(cudaMemcpy(staging.get(), device_nids + offset,
                                      count, cudaMemcpyDeviceToHost));
        std::memcpy(destination + offset, staging.get(), count);
      }
    }
  }

  index_type copy_index_to_dev(const host_index_type& index) {
    index_type dev_index;
    dev_index.nids  = raft_utils::copy_to_dev(index.nids.view(), m_dev_res);
    dev_index.dists = raft_utils::copy_to_dev(index.dists.view(), m_dev_res);
    return dev_index;
  }

  template <typename IndexT>
  index_view_type make_index_view(IndexT& index) const {
    return index_view_type(index.nids.n_rows(), index.nids.n_cols(),
                           const_cast<id_type*>(index.nids.data()),
                           const_cast<dist_type*>(index.dists.data()));
  }

  index_type build_index_object(const point_store_type& pstore,
                                const std::string_view  dist_func,
                                const size_t k, const size_t nnd_k,
                                const double rho, const double delta) {
    if (dist_func != "l2" && dist_func != "cosine") {
      spdlog::critical("Inner product distance is not supported.");
      return index_type{};
    }

    const auto n_points = pstore.n_rows();
    if (n_points == 0 || k == 0) {
      return index_type{};
    }
    if (!pstore.data()) {
      throw std::invalid_argument("Point store is not initialized.");
    }
    if (nnd_k < k) {
      throw std::invalid_argument("nnd_k must be >= k.");
    }

    return priv_build_local_knng(pstore, dist_func, k, nnd_k, rho, delta);
  }

  void add_id_offset(const id_type offset, index_nid_type& inex_nids) {
    if (offset > 0) {
      gpu::add_scalar(inex_nids.data_handle(), offset, inex_nids.size());
    }
  }

  // Allocate point store
  // The ownership of the allocated point store is passed to the caller.
  point_store_type alloc_point_store(const size_t n_rows, const size_t dims) {
    return raft_utils::make_host_matrix<fe_type>(n_rows, dims, m_host_res);
  }

  void allocate_group_index_nids(group_index_nid_type& nids,
                                 const size_t n_rows, const size_t n_cols) {
    nids = raft_utils::make_host_matrix<id_type>(n_rows, n_cols, m_host_res);
  }

  void optimize_index(const index_view_type& index) {
    if (index.size() == 0 || index.num_neighbors() == 0) {
      return;
    }

    if (index.neighbor_ids_data() == nullptr) {
      throw std::invalid_argument(
          "optimize_index requires valid neighbor ID and distance buffers.");
    }

    auto h_cagra_nids_view = raft_utils::make_host_matrix_view<id_type>(
        const_cast<id_type*>(index.neighbor_ids_data()), index.size(),
        index.num_neighbors());
    cuvs::neighbors::cagra::helpers::optimize(m_dev_res, h_cagra_nids_view,
                                              h_cagra_nids_view);
  }

  void update_index(const index_view_type& candidates,
                    const id_type          candidate_id_offset,
                    const id_type master_id_offset, index_type& master_index) {
    const size_t n_rows = master_index.nids.n_rows();
    const size_t k      = master_index.nids.n_cols();
    if (n_rows == 0 || k == 0) {
      return;
    }
    if (candidates.size() == 0 || candidates.num_neighbors() == 0) {
      return;
    }
    if (master_index.nids.data() == nullptr ||
        candidates.neighbor_ids_data() == nullptr) {
      throw std::invalid_argument(
          "update_index requires valid candidate/master neighbor ID buffers.");
    }

    const auto n_updateds = gpu::update_knng(
        master_index.nids.data_handle(), master_index.dists.data_handle(),
        candidates.neighbor_ids_data(), candidates.dists_data(), n_rows, k,
        static_cast<int>(candidates.num_neighbors()), candidate_id_offset,
        master_id_offset, m_dev_res);

    spdlog::trace("Number of updated neighbors: {}", n_updateds);
  }

  index_view_type run_queries(const point_store_view_type& query_pstore,
                              const point_store_view_type& trg_pstore,
                              const index_view_type&       trg_index,
                              const int frontier_size, const int query_k) {
    if (query_k <= 0) {
      throw std::invalid_argument("query_k must be > 0.");
    }
    if (query_pstore.size() == 0 || query_pstore.dims() == 0) {
      return index_view_type();
    }

    priv_run_query_cagra(query_pstore, trg_pstore, trg_index, frontier_size,
                         query_k);
    return index_view_type(
        m_query_result_nids.extent(0), m_query_result_nids.extent(1),
        m_query_result_nids.data_handle(), m_query_result_dists.data_handle());
  }

  void reset() {}

 private:
  index_type priv_build_local_knng(const point_store_type& pstore,
                                   const std::string_view  dist_func,
                                   const size_t k, const size_t nnd_k,
                                   [[maybe_unused]] const double rho,
                                   const double                  delta) {
    using namespace cuvs::neighbors;

    // Prepare for CAGRA search
    if (dist_func == "l2") {
      m_cagra_dist_func = cuvs::distance::DistanceType::L2Expanded;
    } else if (dist_func == "inner_product" || dist_func == "ip") {
      // Memo: cuVS's inner_product returns similarity scores (dot products)
      // where larger values indicate more similar neighbors. This is not
      // compatible with the current pipeline which expects distances where
      // smaller values are better.
      spdlog::critical("Inner product distance is not supported.");
      m_cagra_dist_func = cuvs::distance::DistanceType::InnerProduct;
    } else if (dist_func == "cosine") {
      m_cagra_dist_func = cuvs::distance::DistanceType::CosineExpanded;
    } else {
      std::cerr << "Unsupported distance: " << dist_func << std::endl;
      std::abort();
    }

    cuvs::neighbors::nn_descent::index_params nnd_params(nnd_k,
                                                         m_cagra_dist_func);
    nnd_params.intermediate_graph_degree = nnd_k * 1.5;
    nnd_params.return_distances          = true;
    nnd_params.max_iterations            = nnd_k;
    nnd_params.termination_threshold     = delta;

    rec_time().start("Copy-pstore-to-dev");
    auto d_pstore = raft_utils::copy_to_dev(pstore.view(), m_dev_res);
    rec_time().stop();

    rec_time().start("cuVS-NND-kernel");
    auto nnd_index = cuvs::neighbors::nn_descent::build(
        m_dev_res, nnd_params, raft_utils::make_const_matrix_view(d_pstore));
    rec_time().stop();

    spdlog::trace("Shrink nids");
    rec_time().start("Shrink-nids");
    auto d_nids =
        raft_utils::strided_copy_to_dev(nnd_index.graph(), k, m_dev_res);
    rec_time().stop();

    spdlog::trace("Shrink dists");
    rec_time().start("Shrink-dists");
    const auto d_dists_full = nnd_index.distances();
    if (!d_dists_full) {
      throw std::runtime_error("cuVS NN-descent did not return distances.");
    }
    // nn_descent::index::distances() already resides on the device, unlike
    // graph() which is host memory, so use a device-to-device copy here.
    auto d_dists =
        raft_utils::strided_copy_dev_to_dev(*d_dists_full, k, m_dev_res);
    rec_time().stop();

    return index_type{std::move(d_nids), std::move(d_dists)};
  }

  void priv_run_query_cagra(const point_store_view_type& query_pstore,
                            const point_store_view_type& trg_pstore,
                            const index_view_type&       trg_index,
                            const int frontier_size, const int query_k) {
    spdlog::trace("Copy query data to GPU");
    saltatlas::rec_time().start("Copy-query-data");
    auto local_queries = slntrft::copy_to_dev(
        slntrft::make_host_matrix_view(query_pstore.data(), query_pstore.size(),
                                       query_pstore.dims()),
        m_dev_res);
    auto local_trg_pstore = slntrft::copy_to_dev(
        slntrft::make_host_matrix_view(trg_pstore.data(), trg_pstore.size(),
                                       trg_pstore.dims()),
        m_dev_res);
    saltatlas::rec_time().stop();

    const auto queries   = slntrft::make_const_matrix_view(local_queries);
    const auto n_queries = queries.extent(0);
    const auto local_trg_pstore_view =
        slntrft::make_const_matrix_view(local_trg_pstore);

    if (m_query_result_nids.extent(0) != n_queries ||
        m_query_result_nids.extent(1) != static_cast<size_t>(query_k)) {
      m_query_result_nids =
          slntrft::make_dev_matrix<id_type>(n_queries, query_k, m_dev_res);
      m_query_result_dists =
          slntrft::make_dev_matrix<dist_type>(n_queries, query_k, m_dev_res);
    }

#if 0
    // Diagnostic mode: keep the MPI merge path unchanged while replacing its
    // approximate cross-group search with exact search. This is intentionally
    // opt-in because its cost grows with queries * target points.
    const auto* exact_search = std::getenv("SALTATLAS_SOLANET_EXACT_SEARCH");
    if (exact_search != nullptr && exact_search[0] == '1') {
      if constexpr (std::is_same_v<fe_type, float>) {
        if (!m_logged_exact_search) {
          spdlog::warn("Using exact cross-group search for diagnosis");
          m_logged_exact_search = true;
        }
        cuvs::neighbors::brute_force::index_params index_params;
        index_params.metric = m_cagra_dist_func;
        auto exact_index = cuvs::neighbors::brute_force::build(
            m_dev_res, index_params, local_trg_pstore_view);
        auto exact_nids =
            slntrft::make_dev_matrix<int64_t>(n_queries, query_k, m_dev_res);
        cuvs::neighbors::brute_force::search_params search_params;
        cuvs::neighbors::brute_force::search(
            m_dev_res, search_params, exact_index, queries, exact_nids.view(),
            m_query_result_dists.view());
        gpu::cast_search_ids(m_query_result_nids.data_handle(),
                             exact_nids.data_handle(), exact_nids.size(),
                             m_dev_res);
        raft::resource::sync_stream(m_dev_res);
        return;
      } else {
        throw std::runtime_error(
            "SALTATLAS_SOLANET_EXACT_SEARCH supports float features only.");
      }
    }
#endif

    auto local_trg_index =
        slntrft::copy_to_dev(slntrft::make_host_matrix_view(
                                 trg_index.neighbor_ids_data(),
                                 trg_index.size(), trg_index.num_neighbors()),
                             m_dev_res);
    const auto local_trg_index_view =
        slntrft::make_const_matrix_view(local_trg_index);

    // https://github.com/ROCm-DS/hipVS/blob/release/rocmds-25.10/cpp/include/cuvs/neighbors/cagra.hpp#L175
    cuvs::neighbors::cagra::search_params search_params;
    const int                             effective_frontier_size =
        (frontier_size > 0) ? frontier_size : (query_k * 2);
    search_params.itopk_size           = effective_frontier_size;
    search_params.search_width         = 1;
    search_params.num_random_samplings = 1;
    search_params.team_size            = 8;
    search_params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    search_params.max_iterations = trg_index.size();
    // search_params.min_iterations = search_params.itopk_size * 2;

    // CAGRA Index: dataset and knng
    spdlog::trace("Const CAGRA index");
    saltatlas::rec_time().start("Const-cagra-index");
    using cagra_index_t = cuvs::neighbors::cagra::index<fe_type, id_type>;
    cagra_index_t cagra_index =
        priv_const_cagra_index(local_trg_pstore_view, local_trg_index_view);
    raft::resource::sync_stream(m_dev_res);
    saltatlas::rec_time().stop();

    // spdlog::trace("Alloc query result buffers");
    // auto result_nids = slntrft::make_dev_matrix<id_type>(n_queries, k,
    // m_dev_res); auto result_dists =
    //     slntrft::make_dev_matrix<dist_type>(n_queries, k, m_dev_res);

    spdlog::trace("Search");
    saltatlas::rec_time().start("Search");
    cuvs::neighbors::cagra::search(m_dev_res, search_params, cagra_index,
                                   queries, m_query_result_nids.view(),
                                   m_query_result_dists.view());
    raft::resource::sync_stream(m_dev_res);
    const auto elapsed_sec = saltatlas::rec_time().stop();
    spdlog::trace("Finished searching");
    // Show search performance in QPS (Queries Per Second)
    const double qps = n_queries / elapsed_sec;
    spdlog::trace(
        "Search QPS: #of queries {} / Elapsed time: {:.3f} sec = {} QPS",
        n_queries, elapsed_sec, saltatlas::utility::add_comma_separators(qps));
    const double throughput = trg_index.size() * n_queries / elapsed_sec;
    spdlog::trace(
        "(Search throughput: #of queries: {} X KNNG size: {}) / Elapsed time: "
        "{:.3f} sec = Throughput: {}",
        n_queries, trg_index.size(), elapsed_sec,
        saltatlas::utility::add_comma_separators(throughput));

    // return std::make_pair(std::move(result_nids), std::move(result_dists));
  }

  cagra_index_t priv_const_cagra_index(
      slntrft::d_matrix_view_type<const fe_type> dataset,
      slntrft::d_matrix_view_type<const id_type> knng) {
    try {
      // std::cout << "CAGRA index dataset = " << dataset.extent(0) << " x "
      //           << dataset.extent(1) << std::endl;
      // std::cout << "CAGRA index graph = " << knng.extent(0) << " x "
      //           << knng.extent(1) << std::endl;
      return cagra_index_t(m_dev_res, m_cagra_dist_func, dataset, knng);
    } catch (std::bad_alloc& e) {
      std::cerr << "Insufficient GPU memory to construct CAGRA index with "
                   "dataset on GPU"
                << std::endl;
      std::abort();
    } catch (raft::logic_error& e) {
      std::cerr << "Insufficient GPU memory to construct CAGRA index with "
                   "dataset on GPU"
                << std::endl;
      std::abort();
    }

    // Dummy return value to avoid compiler warning
    return cagra_index_t(m_dev_res);
  }

  raft::resources              m_host_res;
  raft::device_resources       m_dev_res;
  cuvs::distance::DistanceType m_cagra_dist_func;
  bool                         m_verbose{false};
  bool                         m_logged_exact_search{false};

  slntrft::d_matrix_type<id_type>   m_query_result_nids;
  slntrft::d_matrix_type<dist_type> m_query_result_dists;
};

}  // namespace saltatlas::solanet::cuvs_nn

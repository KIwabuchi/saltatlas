// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

// #define PROFILE_FV

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

#include <boost/container/flat_set.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/unordered/unordered_node_map.hpp>
#include <boost/unordered/unordered_node_set.hpp>
#include <metall/metall.hpp>

#include "saltatlas/neo_dnnd/dataset_reader.hpp"
#include "saltatlas/neo_dnnd/distance.hpp"
#include "saltatlas/neo_dnnd/hash.hpp"
#include "saltatlas/neo_dnnd/mpi.hpp"
#include "saltatlas/neo_dnnd/neighbor.hpp"
#include "saltatlas/neo_dnnd/omp.hpp"
#include "saltatlas/neo_dnnd/pfv_replica_store.hpp"
#include "saltatlas/neo_dnnd/point_store.hpp"
#include "saltatlas/neo_dnnd/profile_db.hpp"
#include "saltatlas/neo_dnnd/system.hpp"
#include "saltatlas/neo_dnnd/time.hpp"
#include "saltatlas/neo_dnnd/trace_event.hpp"
#include "saltatlas/neo_dnnd/utility.hpp"

#define FV_SEND_BATCH_SIZE_BYTE (1ULL << 27)

namespace saltatlas::neo_dnnd {

namespace {
namespace bst   = boost;
namespace bstc  = bst::container;
namespace bstuo = bst::unordered;
namespace d2dtl = saltatlas::neo_dnnd::ndndtl;
}  // namespace

// TODO: change the location of the following functions
template <typename id_type>
inline int partition(const id_type id, const int comm_size) {
  return hash<826>()(id) % comm_size;
}

template <typename _id_type, typename _feature_element_type,
          typename _distance_type>
class dnnd_kernel {
 public:
  using id_type              = _id_type;
  using feature_element_type = _feature_element_type;
  using distance_type        = _distance_type;
  using point_store          = saltatlas::neo_dnnd::point_store<
               id_type, feature_element_type,
               metall::manager::allocator_type<std::byte>>;
  using distance_function =
      distance::similarity_type<feature_element_type, distance_type>;

  static_assert(sizeof(std::pair<id_type, id_type>) == 2 * sizeof(id_type),
                "sizeof(std::pair<id_type, id_type>) != 2 * sizeof(id_type)");

 private:
  struct __attribute__((packed)) id_pair_type {
    id_pair_type() = default;
    id_pair_type(const id_type first, const id_type second)
        : first(first), second(second) {}

    bool operator==(const id_pair_type &other) const {
      return first == other.first && second == other.second;
    }

    id_type first;
    id_type second;
  };

  struct __attribute__((packed)) nb_dist_type {
    nb_dist_type() = default;
    nb_dist_type(const id_type first, const id_type second,
                 const distance_type dist)
        : first(first), second(second), distance(dist) {}

    id_type       first;
    id_type       second;
    distance_type distance;
  };

  struct id_pair_hasher {
    std::size_t operator()(const id_pair_type &id_pair) const {
      std::size_t buf;
      ndndtl::murmurhash3::MurmurHash3_x64_128(&id_pair, sizeof(id_pair), 12321,
                                               &buf);
      return buf;
    }
  };

  using knn_list_t = d2dtl::knn_list<id_type, distance_type, bool>;
  // This is for internal use only. Output knng is another type.
  using knng_heap_t = bstuo::unordered_flat_map<id_type, knn_list_t>;

  template <typename T>
  using adj_list = bstuo::unordered_flat_map<id_type, std::vector<T>>;

  template <typename T>
  using matrix2d = std::vector<std::vector<T>>;

  using pop_fv_cache_t =
      ndndtl::pfv_replica_store<id_type, feature_element_type>;

  // For macOS
#ifdef __APPLE__
  static constexpr std::size_t k_def_cache_capacity = 1 << 10;
  static constexpr const char *k_shm_dir            = "/tmp/";
#else
  static constexpr std::size_t       k_def_cache_capacity = 1 << 18;
  static constexpr const char *const k_shm_dir            = "/dev/shm/";
#endif
  static constexpr const char *k_point_store_shm_name = "pstore";

 public:
  using neighbor_type = typename knn_list_t::neighbor_type;
  using knng_type =
      bstuo::unordered_flat_map<id_type, std::vector<neighbor_type>>;

  dnnd_kernel(distance_function  &distance_func,
              time_recorder_base &time_recorder, mpi::communicator &comm,
              const bool         verbose  = false,
              const unsigned int rnd_seed = std::random_device{}())
      : m_distance_func(distance_func),
        m_time_recorder(time_recorder),
        m_comm(comm),
        m_verbose(verbose),
        m_rng(rnd_seed + m_comm.rank()),
        m_numa_separate_node(false) {
    m_all_to_all_pairs =
        mpi::get_pair_wise_all_to_all_pattern(m_comm.size(), m_comm.rank());
    m_all_to_all_region_pairs = mpi::get_pair_wise_all_to_all_pattern(
        priv_num_regions(), priv_region_no());
  }

  ~dnnd_kernel() noexcept {
    priv_free_mpi_types();
    m_point_stores.clear();
    for (auto *manager : m_point_store_managers) {
      delete manager;
    }
    m_comm.barrier();
  }

  void read_dataset(const std::filesystem::path &dataset_path,
                    const std::string_view      &dataset_format,
                    const bool                   share_pstore_regionally) {
    m_share_pstore_regionally = share_pstore_regionally;
    priv_cout0(m_verbose) << "Share point store regionally: "
                          << m_share_pstore_regionally << std::endl;

    auto partitioner = [this](const id_type id) {
      return saltatlas::neo_dnnd::partition(id, m_comm.size());
    };
    const auto points = saltatlas::neo_dnnd::dataset_reader<
        id_type, feature_element_type>::read(dataset_path, dataset_format,
                                             partitioner, m_comm);
    m_comm.barrier();

    const std::size_t num_points = points.size();
    if (num_points == 0) {
      // Current implementation requires at least one point to be assigned to
      // each MPI rank.
      std::cerr << m_comm.rank() << ": No points are assigned." << std::endl;
      m_comm.abort();
    }

    id_type max_id = 0;
    for (const auto &[id, fv] : points) {
      max_id = std::max(max_id, id);
    }
    max_id = m_comm.all_reduce_max(max_id);
    priv_cout0(m_verbose) << "Max ID: " << max_id << std::endl;
    assert(max_id + 1 == m_comm.all_reduce_sum(points.size()));
    m_num_total_points = max_id + 1;

    auto gen_pstore_name = [](const int rank) {
      std::string name = k_shm_dir;
      name += "/";
      name += k_point_store_shm_name;
      name += "-";
      name += std::to_string(rank);
      return name;
    };

    m_comm.cout0() << "\nIngesting to point store..." << std::endl;
    m_num_dims = (num_points > 0) ? points.front().second.size() : 0;
    {
      const std::string path = gen_pstore_name(m_comm.rank());
      metall::manager   manager(metall::create_only, path);
      auto *pstore = manager.construct<point_store>(metall::unique_instance)(
          manager.get_allocator());
      pstore->init(num_points, m_num_dims);
      for (const auto &[id, fv] : points) {
        std::memcpy((*pstore)[id], fv.data(),
                    fv.size() * sizeof(feature_element_type));
      }
    }
    m_comm.barrier();
    m_comm.cout0() << "Finished constructing point store" << std::endl;

    priv_cout0(m_verbose) << "Opening shared point stores" << std::endl;
    {
      m_point_store_managers.clear();
      m_point_stores.clear();
      for (std::size_t r = 0; r < priv_region_size(); ++r) {
        if (!m_share_pstore_regionally && r != priv_regional_rank()) {
          m_point_store_managers.emplace_back(nullptr);
          m_point_stores.emplace_back(nullptr);
          continue;
        }

        const int         rank = r + priv_region_no() * priv_region_size();
        const std::string path = gen_pstore_name(rank);

        m_point_store_managers.emplace_back(
            new metall::manager(metall::open_read_only, path));
        assert(m_point_store_managers.back());
        auto *pstore = m_point_store_managers.back()
                           ->find<point_store>(metall::unique_instance)
                           .first;
        assert(pstore);
        m_point_stores.emplace_back(pstore);
      }
    }
    m_comm.barrier();
  }

  std::size_t num_dims() const { return m_num_dims; }

  knng_type construct(const std::size_t k, const double rho = 0.8,
                      const double      delta                = 0.001,
                      const bool        remove_duplicate_fvs = true,
                      const std::size_t batch_size           = 1 << 20,
                      const double      popular_fv_ratio     = 0.0) {
    {
      m_fv_send_batch_size =
          FV_SEND_BATCH_SIZE_BYTE / num_dims() / sizeof(feature_element_type);
      m_fvs_disp.resize(m_fv_send_batch_size);
      m_fvs_block_lengths.resize(m_fv_send_batch_size, 1);
      priv_commit_mpi_types();
    }
    if (m_verbose) {
      priv_show_dram_usage();
    }
    priv_cache_popular_fvs(popular_fv_ratio);
    if (m_verbose) {
      priv_show_dram_usage();
    }

    m_k                    = k;
    m_rho                  = rho;
    m_delta                = delta;
    m_remove_duplicate_fvs = remove_duplicate_fvs;

    assert(m_num_total_points > 0);
    priv_cout0(m_verbose) << "Terminal threshold: "
                          << m_delta * m_k * m_num_total_points << std::endl;
    if (m_num_total_points - 1 < m_k) {
      m_comm.cerr0() << "ERROR: there are not enough points to construct a kNNG"
                     << std::endl;
      m_comm.abort();
    }

    priv_cout0(m_verbose) << std::endl;
    priv_cout0(true) << "Initialize graph" << std::endl;
    m_time_recorder.start("Init graph");
    priv_init_graph();
    m_time_recorder.stop();

    priv_cout0(m_verbose) << std::endl;
    priv_cout0(true) << "Start NN-Descent core loop" << std::endl;
    while (true) {
      priv_cout0(m_verbose) << "\n--------------------" << std::endl;
      priv_cout0(m_verbose)
          << "Superstep No.: " << m_super_step_no << std::endl;
      priv_cout0(m_verbose) << "--------------------" << std::endl;

      adj_list<id_type> old_ng;
      adj_list<id_type> new_ng;
      m_time_recorder.start("Set old and new");
      priv_set_old_and_new(old_ng, new_ng);
      m_time_recorder.stop_and_report(priv_cout0(m_verbose));
      priv_cout0(m_verbose) << std::endl;

      m_time_recorder.start("Add reversed edges");
      priv_add_reverse_neighbors(old_ng, new_ng);
      m_time_recorder.stop_and_report(priv_cout0(m_verbose));
      priv_cout0(m_verbose) << std::endl;

      m_time_recorder.start("Neighbor checks");
      const auto num_updated =
          priv_gen_and_launch_neighbor_checks(old_ng, new_ng, batch_size);
      priv_cout0(m_verbose) << std::endl;
      m_time_recorder.stop_and_report(priv_cout0(m_verbose));
      priv_cout0(m_verbose) << "#of updates: " << num_updated << std::endl;

      if (num_updated < m_delta * m_k * m_num_total_points) {
        break;
      }
      ++m_super_step_no;
    }
    m_comm.cout0() << "Finished NN-Descent core loop" << std::endl;
    m_comm.cout0() << std::endl;
    m_comm.barrier();

    priv_cout0(m_verbose) << std::endl;
    m_time_recorder.start("Convert KNNG");
    knng_type knng;
    knng.reserve(m_graph.size());
    for (auto &elem : m_graph) {
      const auto sid = elem.first;
      auto      &nn  = elem.second;
      knng[sid]      = nn.extract_neighbors();
    }
    m_comm.barrier();
    m_time_recorder.stop_and_report(priv_cout0(m_verbose));

    return knng;
  }

  /// \brief Optimize kNNG. Specifically, make the graph undirected and prune
  /// high-degree vertices. If m < 0, no pruning is performed.
  void optimize(const double m, knng_type &knng) {
    m_comm.cout0() << "Start optimization" << std::endl;
    priv_cout0(m_verbose) << "m: " << m << std::endl;

    matrix2d<id_type>                       r_nids(m_comm.size());
    std::vector<std::vector<distance_type>> r_dits(m_comm.size());

    for (const auto &item : knng) {
      const auto  sid       = item.first;
      const auto &neighbors = item.second;
      for (const auto &neighbor : neighbors) {
        const auto nid = neighbor.id;
        const auto dit = neighbor.distance;
        r_nids[priv_owner(nid)].push_back(nid);
        r_nids[priv_owner(nid)].push_back(sid);
        r_dits[priv_owner(nid)].push_back(dit);
      }
    }

    m_comm.barrier();

    for (const auto pair_rank : m_all_to_all_pairs) {
      std::vector<id_type> recv_nids;
      {
        const auto send_buf = std::move(r_nids[pair_rank]);
        m_comm.sendrecv_arb_size(pair_rank, send_buf, recv_nids);
      }

      std::vector<distance_type> recv_dists;
      {
        const auto send_buf = std::move(r_dits[pair_rank]);
        m_comm.sendrecv_arb_size(pair_rank, send_buf, recv_dists);
      }

      for (std::size_t i = 0; i < recv_nids.size(); i += 2) {
        const auto r_sid = recv_nids[i];
        const auto r_nid = recv_nids[i + 1];
        assert(priv_owner(r_sid) == m_comm.rank());
        knng[r_sid].push_back(neighbor_type(r_nid, recv_dists[i / 2]));
      }
    }
    m_comm.barrier();

    for (auto &item : knng) {
      auto &neighbors = item.second;
      std::sort(
          neighbors.begin(), neighbors.end(),
          [](const auto &x, const auto &y) { return x.distance < y.distance; });

      // remove duplicates
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end(),
                                  [](const auto &lhd, const auto &rhd) {
                                    return lhd.id == rhd.id;
                                  }),
                      neighbors.end());

      // Prune high-degree vertices
      if (m < 0) {  // No pruning
        continue;
      }
      if (neighbors.size() > m_k * m) {
        neighbors.resize(m_k * m);
      }
    }
    m_comm.barrier();
  }

  void print_profile([[maybe_unused]] const bool final) const {
    if (final || m_verbose) {
      m_comm.cout0() << "FV processing breakdown:" << std::endl;
      m_comm.cout0() << "  #of normally sent:\t"
                     << (std::size_t)m_comm.all_reduce_sum(
                            m_profile_db.get_total("SentFVs"))
                     << std::endl;

      m_comm.cout0() << "  #of FVs in node-local shared FV store:\t"
                     << (std::size_t)m_comm.all_reduce_sum(
                            m_profile_db.get_total("RegionallyCheckedFVs"))
                     << std::endl;

      m_comm.cout0() << "  #of removed duplicates FVs:\t"
                     << (std::size_t)m_comm.all_reduce_sum(
                            m_profile_db.get_total("DupFVs"))
                     << std::endl;

      m_comm.cout0() << "  #of FVs in replicate store:\t"
                     << (std::size_t)m_comm.all_reduce_sum(
                            m_profile_db.get_total("ReplicatedFVs"))
                     << std::endl;
    }

#ifdef PROFILE_FV
    if (final) {
      // make histogram
      std::size_t max_val = 0;
      for (const auto &item : m_fv_count) {
        max_val = std::max(max_val, item.second);
      }
      max_val = m_comm.all_reduce_max(max_val);

      if (max_val == 0) {
        m_comm.cout0() << "FV count histogram: No data" << std::endl;
        return;
      }

      const std::size_t num_bins = std::size_t(std::log2l(double(max_val))) + 1;
      std::vector<std::size_t> hist(num_bins, 0);
      for (const auto &item : m_fv_count) {
        const auto bin = std::size_t(std::log2l(double(item.second)));
        hist[bin] += 1;
      }
      const auto global_hist = m_comm.reduce_sum(hist, 0);
      m_comm.cout0() << "FV count histogram:" << std::endl;
      for (std::size_t i = 0; i < global_hist.size(); ++i) {
        m_comm.cout0() << "[ " << (std::size_t)std::pow((double)2, (double)i)
                       << " , "
                       << (std::size_t)std::pow((double)2, (double)i + 1)
                       << " ) \t" << global_hist[i] << std::endl;
      }
    }
#endif
    if (m_verbose) {
      priv_show_dram_usage();
    }
  }

 private:
  std::ostream &priv_cout0(const bool verbose) const {
    static std::ostringstream dummy;
    if (verbose) {
      return m_comm.cout0();
    } else {
      return dummy;
    }
  }

  int priv_owner(const id_type id) const {
    return partition(id, m_comm.size());
  }

  inline void priv_show_dram_usage() const {
    const ssize_t total_ram     = system::get_total_ram_size();
    const ssize_t used_ram      = system::get_used_ram_size();
    const ssize_t free_ram      = system::get_free_ram_size();
    const ssize_t page_cache    = system::get_page_cache_size();
    const ssize_t available_ram = system::get_available_ram_size();

    auto show_procedure = [&](const ssize_t val, const std::string &name) {
      const auto val_gb = val / double(1ULL << 30);
      // show min, max, mean, total
      const auto min_val = m_comm.all_node_reduce_min(val_gb);
      const auto max_val = m_comm.all_node_reduce_max(val_gb);
      const auto total   = m_comm.all_node_reduce_sum(val_gb);
      const auto mean    = total / m_comm.num_nodes();
      m_comm.cout0() << name << ":\t";
      m_comm.cout0() << std::setprecision(3) << min_val << "\t" << max_val
                     << "\t" << mean << "\t" << total << std::endl;
    };

    m_comm.cout0() << "DRAM usage GB (min, max, mean, total):" << std::endl;
    show_procedure(total_ram, "TotalRAM");
    show_procedure(used_ram, "UsedRAM");
    show_procedure(free_ram, "FreeRAM");
    show_procedure(page_cache, "PageCache");
    show_procedure(available_ram, "AvailableRAM");
  }

  void priv_commit_mpi_types() {
    priv_free_mpi_types();
    {
      int          blocklengths[3] = {1, 1, 1};
      MPI_Datatype types[3]        = {mpi::data_type::get<id_type>(),
                                      mpi::data_type::get<id_type>(),
                                      mpi::data_type::get<distance_type>()};
      MPI_Aint     offsets[3];
      offsets[0] = offsetof(nb_dist_type, first);
      offsets[1] = offsetof(nb_dist_type, second);
      offsets[2] = offsetof(nb_dist_type, distance);
      MPI_Type_create_struct(3, blocklengths, offsets, types, &m_nb_dist_type);
      MPI_Type_commit(&m_nb_dist_type);
    }

    {
      DNND2_CHECK_MPI(::MPI_Type_contiguous(2, mpi::data_type::get<id_type>(),
                                            &m_id_pair_type));
      DNND2_CHECK_MPI(::MPI_Type_commit(&m_id_pair_type));
    }

    {
      DNND2_CHECK_MPI(::MPI_Type_contiguous(
          num_dims(), mpi::data_type::get<feature_element_type>(),
          &m_feature_type));
      DNND2_CHECK_MPI(::MPI_Type_commit(&m_feature_type));
    }

    { m_fvs_types.resize(m_fv_send_batch_size, m_feature_type); }
  }

  void priv_free_mpi_types() {
    if (m_nb_dist_type != MPI_DATATYPE_NULL) {
      DNND2_CHECK_MPI(::MPI_Type_free(&m_nb_dist_type));
      m_nb_dist_type = MPI_DATATYPE_NULL;
    }
    if (m_id_pair_type != MPI_DATATYPE_NULL) {
      DNND2_CHECK_MPI(::MPI_Type_free(&m_id_pair_type));
      m_id_pair_type = MPI_DATATYPE_NULL;
    }
    if (m_feature_type != MPI_DATATYPE_NULL) {
      DNND2_CHECK_MPI(::MPI_Type_free(&m_feature_type));
      m_feature_type = MPI_DATATYPE_NULL;
    }
  }

  void priv_init_graph() {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    // init m_graph space
    for (auto id_itr = point_store.ids_begin(); id_itr != point_store.ids_end();
         ++id_itr) {
      const auto sid = *id_itr;
      m_graph[sid].init(m_k);
    }

    // hold initial graph data grouped by neighbors' owners
    matrix2d<std::pair<id_type, id_type>> init_neighbors_table;
    init_neighbors_table.resize(m_comm.size());

    id_type local_max_id = 0;
    for (auto id_itr = point_store.ids_begin(); id_itr != point_store.ids_end();
         ++id_itr) {
      local_max_id = std::max(local_max_id, *id_itr);
    }
    const auto max_point_id = m_comm.all_reduce_max(local_max_id);
    priv_cout0(m_verbose) << "Max point ID: " << max_point_id << std::endl;

    for (auto id_itr = point_store.ids_begin(); id_itr != point_store.ids_end();
         ++id_itr) {
      const auto sid = *id_itr;
      assert(priv_owner(sid) == m_comm.rank());

      bstc::flat_set<id_type> used;
      for (std::size_t k = 0; k < m_k; ++k) {
        id_type nid = -1;
        do {
          nid = std::uniform_int_distribution<id_type>(0, max_point_id)(m_rng);
        } while (nid == sid || used.count(nid) > 0);
        init_neighbors_table[priv_owner(nid)].emplace_back(sid, nid);
        used.insert(nid);
      }
    }

    std::vector<id_type>              recv_buf;
    std::vector<feature_element_type> feature_recv_buf;
    for (std::size_t ri = 0; ri < m_all_to_all_pairs.size(); ++ri) {
      const auto pair_rank      = m_all_to_all_pairs[ri];
      auto      &init_neighbors = init_neighbors_table[pair_rank];

      // Send feature vector requests to the pair rank
      std::vector<id_type> send_buf;
      for (const auto &elem : init_neighbors) {
        send_buf.push_back(elem.second);
      }
      m_comm.sendrecv_arb_size(pair_rank, send_buf, recv_buf);

      // Pack requested feature vectors and send them back to the pair rank.
      std::vector<feature_element_type> feature_send_buf;
      feature_send_buf.resize(recv_buf.size() * num_dims());
      for (std::size_t i = 0; i < recv_buf.size(); ++i) {
        const auto id = recv_buf[i];
        assert(priv_owner(id) == m_comm.rank());
        const auto *feature = point_store[id];
        std::copy(feature, feature + num_dims(),
                  feature_send_buf.begin() + i * num_dims());
      }
      m_comm.sendrecv_arb_size(pair_rank, feature_send_buf, feature_recv_buf);

      // calculate distances and update graph
      std::size_t buf_i = 0;
      for (const auto &elem : init_neighbors) {
        const auto sid = elem.first;
        const auto nid = elem.second;

        assert(priv_owner(sid) == m_comm.rank());
        const auto dist =
            m_distance_func(point_store[sid], num_dims(),
                            &feature_recv_buf[buf_i * num_dims()], num_dims());
        m_graph[sid].push(nid, dist, true);  // push as a new neighbor
        ++buf_i;
      }
      init_neighbors.clear();  // Reduce memory usage
    }

    m_comm.barrier();
  }

  void priv_set_old_and_new(adj_list<id_type> &old_ng,
                            adj_list<id_type> &new_ng) {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    old_ng.clear();
    new_ng.clear();
    old_ng.reserve(point_store.num_points());
    new_ng.reserve(point_store.num_points());

    std::size_t num_olds = 0;
    std::size_t num_news = 0;
    for (auto id_itr = point_store.ids_begin(); id_itr != point_store.ids_end();
         ++id_itr) {
      const auto sid     = *id_itr;
      auto      &old_nbs = old_ng[sid];
      auto      &new_nbs = new_ng[sid];
      old_nbs.clear();
      new_nbs.clear();

      for (const auto &n : m_graph[sid]) {
        const auto  nid  = n.first;
        const auto &flag = n.second;
        if (flag) {
          new_nbs.push_back(nid);  // sample news later
        } else {
          old_nbs.push_back(nid);
          ++num_olds;
        }
      }

      // Sample new neighbors
      std::shuffle(new_nbs.begin(), new_nbs.end(), m_rng);
      const std::size_t num_samples =
          std::min<std::size_t>(m_k * m_rho, new_nbs.size());
      new_nbs.resize(num_samples);
      num_news += num_samples;
      for (const auto &new_id : new_nbs) {
        m_graph[sid].value(new_id) = false;  // mark as old
      }
    }

    priv_cout0(m_verbose) << "#of olds: " << m_comm.all_reduce_sum(num_olds)
                          << std::endl;
    priv_cout0(m_verbose) << "#of news: " << m_comm.all_reduce_sum(num_news)
                          << std::endl;
  }

  void priv_add_reverse_neighbors(adj_list<id_type> &old_ng,
                                  adj_list<id_type> &new_ng) {
    priv_cout0(m_verbose) << "Adding reversed old neighbors" << std::endl;
    priv_add_reverse_neighbors(old_ng);
    priv_cout0(m_verbose) << "Adding reversed new neighbors" << std::endl;
    priv_add_reverse_neighbors(new_ng);
  }

  void priv_add_reverse_neighbors(adj_list<id_type> &ng) {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    // generate reversed neighbors, grouping by the owner of the neighbor

    m_time_recorder.start("Gen r-NBRs locally");
    matrix2d<id_pair_type> r_ng(m_comm.size());
    for (auto id_itr = point_store.ids_begin(); id_itr != point_store.ids_end();
         ++id_itr) {
      const auto  sid     = *id_itr;
      const auto &nb_list = ng.at(sid);
      for (const auto &nid : nb_list) {
        r_ng[priv_owner(nid)].emplace_back(nid, sid);
      }
    }
    m_time_recorder.stop();
    m_comm.barrier();

    // receive reversed neighbors, removing duplicates
    bstuo::unordered_node_map<id_type, bstc::flat_set<id_type>> r_nb_recv_table;
    r_nb_recv_table.reserve(point_store.num_points());

    m_time_recorder.start("Exchange r-NBRs");
    std::vector<id_pair_type> recv_buf;
    for (std::size_t ri = 0; ri < m_all_to_all_pairs.size(); ++ri) {
      const auto pair_rank = m_all_to_all_pairs[ri];
      // Move data to reduce memory consumption
      auto send_buf = std::move(r_ng[pair_rank]);
      m_comm.sendrecv_arb_size_opt(pair_rank, send_buf, recv_buf,
                                   m_id_pair_type);
      for (std::size_t i = 0; i < recv_buf.size(); ++i) {
        const auto sid = recv_buf[i].first;
        const auto nid = recv_buf[i].second;
        assert(priv_owner(sid) == m_comm.rank());
        r_nb_recv_table[sid].insert(nid);
      }
    }
    m_time_recorder.stop();
    m_comm.barrier();

    m_time_recorder.start("Add r-NBRs");
    // Add reversed edges
    std::size_t max_size = 0;
    for (auto &elem : r_nb_recv_table) {
      auto &r_nbs = elem.second.get_sequence_ref();

      // Remove duplicates
      std::sort(r_nbs.begin(), r_nbs.end());
      r_nbs.erase(std::unique(r_nbs.begin(), r_nbs.end()), r_nbs.end());

      // Sample reversed neighbors
      std::shuffle(r_nbs.begin(), r_nbs.end(), m_rng);
      const std::size_t num_to_select =
          std::min<std::size_t>(m_k * m_rho, r_nbs.size());
      r_nbs.resize(num_to_select);

      // Add reversed neighbors
      auto &nbs = ng[elem.first];
      nbs.insert(nbs.end(), r_nbs.begin(), r_nbs.end());

      // Remove duplicates
      std::sort(nbs.begin(), nbs.end());
      nbs.erase(std::unique(nbs.begin(), nbs.end()), nbs.end());
      max_size = std::max(max_size, nbs.size());
    }
    m_time_recorder.stop();
    priv_cout0(m_verbose) << "Max #of neighbors: "
                          << m_comm.all_reduce_max(max_size) << std::endl;
  }

  std::size_t priv_gen_and_launch_neighbor_checks(
      const adj_list<id_type> &old_ng, const adj_list<id_type> &new_ng,
      const std::size_t batch_size) {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));
    matrix2d<id_pair_type> neighbor_intros(m_comm.size());

    std::size_t batch_no    = 0;
    std::size_t num_updates = 0;
    // flag to check if all checks have been performed by this rank
    bool finished_all_local = false;

    // launch the neighbor check procedure and return the number of the ranks
    // that finished with all checks.
    auto launch_check_procedure = [&]() {
      priv_cout0(m_verbose) << "\nBatch No.: " << batch_no << std::endl;

      ++batch_no;

      num_updates += priv_launch_neighbor_checks(neighbor_intros);
      for (auto &item : neighbor_intros) {
        item.clear();
      }

      print_profile(false);

      const auto num_finished_ranks =
          m_comm.all_reduce_sum(int(finished_all_local));
      return num_finished_ranks;
    };

    // new-new neighbor checks
    std::size_t cnt_checks = 0;
    while (true) {
      if (!finished_all_local) {
        for (auto id_itr = point_store.ids_begin();
             id_itr != point_store.ids_end(); ++id_itr) {
          const auto sid = *id_itr;
          if (new_ng.count(sid) == 0) continue;

          // Generate new-new neighbor checks
          const auto &nns = new_ng.at(sid);
          for (const auto &x : nns) {
            for (const auto &y : nns) {
              if (x >= y) continue;

              neighbor_intros[priv_owner(x)].emplace_back(x, y);
              ++cnt_checks;
              if (cnt_checks % batch_size == 0) {
                launch_check_procedure();
              }
            }
          }
        }
      }
      finished_all_local = true;
      if (launch_check_procedure() == m_comm.size()) {
        break;
      }
    }

    priv_cout0(m_verbose) << "#of performed new-new checks: "
                          << m_comm.all_reduce_sum(cnt_checks) << std::endl;

    // old-new neighbor checks
    cnt_checks         = 0;
    finished_all_local = false;
    while (true) {
      if (!finished_all_local) {
        for (auto id_itr = point_store.ids_begin();
             id_itr != point_store.ids_end(); ++id_itr) {
          const auto sid = *id_itr;
          if (new_ng.count(sid) == 0) continue;

          for (const auto &nid : new_ng.at(sid)) {
            if (old_ng.count(sid) == 0) continue;
            for (const auto &oid : old_ng.at(sid)) {
              // As new and old contain reversed neighbors, new and old could
              // have the same neighbors.
              if (nid == oid) {
                continue;
              }

              if (nid < oid) {
                neighbor_intros[priv_owner(nid)].emplace_back(nid, oid);
              } else {
                neighbor_intros[priv_owner(oid)].emplace_back(oid, nid);
              }

              ++cnt_checks;
              if (cnt_checks % batch_size == 0) {
                launch_check_procedure();
              }
            }
          }
        }
      }
      finished_all_local = true;
      if (launch_check_procedure() == m_comm.size()) {
        break;
      }
    }
    priv_cout0(m_verbose) << "#of performed old-new checks: "
                          << m_comm.all_reduce_sum(cnt_checks) << std::endl;

    return m_comm.all_reduce_sum(num_updates);
  }

  std::size_t priv_launch_neighbor_checks(
      const matrix2d<id_pair_type> &neighbor_intros) {
    matrix2d<id_pair_type> neighbor_checks(m_comm.size());
    matrix2d<id_pair_type> cached_neighbor_checks(m_comm.size());
    assert(int(neighbor_intros.size()) == m_comm.size());

    // Send neighbor introductions
    m_time_recorder.start("Send neighbor intros");
    priv_exchange_neighbor_intros(neighbor_intros, neighbor_checks,
                                  cached_neighbor_checks);
    const auto time_send_ncks = m_time_recorder.stop();
    m_comm.barrier();

    priv_cout0(m_verbose) << "Send nb-intros took (s):\t"
                          << m_comm.all_reduce_max(time_send_ncks) << std::endl;

    // Neighbor checks between pairs
    m_time_recorder.start("Neighbor checks core");
    const auto num_updated = priv_check_neighbors(
        std::move(neighbor_checks), std::move(cached_neighbor_checks));
    const auto time_nck_core = m_time_recorder.stop();
    m_comm.barrier();

    priv_cout0(m_verbose) << "Neighbor check core took (s):\t"
                          << m_comm.all_reduce_max(time_nck_core) << std::endl;

    return num_updated;
  }

  void priv_exchange_neighbor_intros(
      const matrix2d<id_pair_type> &nb_intros,
      matrix2d<id_pair_type>       &recv_nb_chks,
      matrix2d<id_pair_type>       &recv_cached_nb_chks) {
    std::size_t num_cached_checks = 0;
    for (std::size_t ri = 0; ri < m_all_to_all_pairs.size(); ++ri) {
      const auto pair_rank = m_all_to_all_pairs[ri];
      // send requests to the pair rank
      auto &send_buf = nb_intros.at(pair_rank);

      m_time_recorder.start("Send NCKS");
      std::vector<id_pair_type> recv_buf;
      m_comm.sendrecv_arb_size(pair_rank, send_buf, recv_buf, m_id_pair_type);
      m_time_recorder.stop();

      m_time_recorder.start("Unpack NCKS");
      // Assign check suggestions based on the owner of the nid.
      for (std::size_t i = 0; i < recv_buf.size(); ++i) {
        const auto sid = recv_buf[i].first;
        const auto nid = recv_buf[i].second;
        assert(priv_owner(sid) == m_comm.rank());
        if (nid < m_min_cache_id) {
          recv_nb_chks[priv_owner(nid)].emplace_back(sid, nid);
        } else {
          recv_cached_nb_chks[priv_owner(nid)].emplace_back(sid, nid);
          ++num_cached_checks;
        }
      }
      m_time_recorder.stop();
#if 0
      m_time_recorder.start("Polish NCKS");
      for (auto &checks : recv_nb_chks) {
        // Remove duplicates
        bstuo::unordered_flat_set<id_pair_type, id_pair_hasher> set;
        set.insert(checks.begin(), checks.end());
        m_profile_db.add("Dup NCKS", checks.size() - set.size());

        checks.assign(set.begin(), set.end());
        // Sort checks, first key is sid, second key is nid
        std::sort(checks.begin(), checks.end(),
                  [](const auto &x, const auto &y) {
                    return x.first < y.first ||
                           (x.first == y.first && x.second < y.second);
                  });
      }
      m_time_recorder.stop();
#endif
    }
    m_profile_db.add("ReplicatedFVs", num_cached_checks);
  }

  std::size_t priv_check_neighbors(
      matrix2d<id_pair_type> &&neighbor_checks,
      matrix2d<id_pair_type> &&cached_neighbor_checks) {
    // Construct the following containers here to reuse memory over iterations
    std::vector<feature_element_type> recv_features;
    std::vector<std::size_t>          fv_indices_recv;
    std::vector<id_pair_type>         received_checks;
    std::vector<distance_type>        distances;
    std::vector<distance_type>        recv_distances;

    std::size_t num_updated = 0;

    for (std::size_t ri = 0; ri < m_all_to_all_pairs.size(); ++ri) {
      const auto pair_rank       = m_all_to_all_pairs[ri];
      auto       assigned_checks = std::move(neighbor_checks[pair_rank]);

      m_time_recorder.start("Replicated neighbor check");
      num_updated += priv_check_cached_neighbors(
          pair_rank, std::move(cached_neighbor_checks[pair_rank]));
      m_time_recorder.stop();

      if (m_share_pstore_regionally &&
          priv_region_no(pair_rank) == priv_region_no()) {
        // Share the point store with the pair rank
        const auto &my_pstore = *(m_point_stores.at(priv_regional_rank()));
        const auto &pair_pstore =
            *(m_point_stores.at(priv_regional_rank(pair_rank)));

        m_profile_db.add("RegionallyCheckedFVs", assigned_checks.size());

        m_time_recorder.start("Calc dist (regional)");
        distances.resize(assigned_checks.size());
        OMP_DIRECTIVE(parallel for)
        for (std::size_t c = 0; c < assigned_checks.size(); ++c) {
          const auto &[src, nb] = assigned_checks[c];
          assert(priv_owner(src) == m_comm.rank());
          assert(priv_owner(nb) == pair_rank);
          const auto *src_fv = my_pstore[src];
          const auto *nb_fv  = pair_pstore[nb];
          const auto  dist =
              m_distance_func(src_fv, num_dims(), nb_fv, num_dims());
          distances[c] = dist;
        }
        m_time_recorder.stop();

        MPI_Request send_checks_req;
        m_comm.isend(assigned_checks.data(), assigned_checks.size(),
                     m_id_pair_type, pair_rank, send_checks_req);

        MPI_Request send_distances_req;
        m_comm.isend(distances, pair_rank, send_distances_req);

        std::vector<id_pair_type> recv_checks;
        MPI_Request               recv_checks_req;
        m_comm.irecv(pair_rank, recv_checks, m_id_pair_type, recv_checks_req);

        MPI_Request recv_distances_req;
        m_comm.irecv(pair_rank, recv_distances, recv_distances_req);

        m_time_recorder.start("Update kNNG sender (regional)");
        num_updated += priv_update_knng(assigned_checks, false, distances);
        m_time_recorder.stop();

        m_time_recorder.start("Wait for CHKs&DISTs (regional)");
        m_comm.wait(recv_checks_req);
        m_comm.wait(recv_distances_req);
        m_time_recorder.stop();

        m_time_recorder.start("Update kNNG receiver (regional)");
        num_updated += priv_update_knng(recv_checks, true, recv_distances);
        m_time_recorder.stop();

        m_comm.wait(send_checks_req);
        m_comm.wait(send_distances_req);
      } else {
        // Send neighbor checks to the pair rank
        m_time_recorder.start("Send NCK");
        m_comm.sendrecv_arb_size(pair_rank, assigned_checks, received_checks,
                                 m_id_pair_type);
        m_time_recorder.stop();

        // Send feature vectors to the pair rank.
        m_time_recorder.start("Send FVs");
        priv_send_fvs(assigned_checks, pair_rank, recv_features,
                      fv_indices_recv);
        m_time_recorder.stop();

#ifdef PROFILE_FV
        for (const auto &[src, dst] : assigned_checks) {
          if (m_fv_count.count(dst) == 0) {
            m_fv_count[dst] = 0;
          }
          m_fv_count[dst] += 1;
        }
#endif
        // calculate distances
        m_time_recorder.start("Calc dist");
        priv_cal_distances(received_checks, recv_features, fv_indices_recv,
                           distances);
        m_time_recorder.stop();

        ::MPI_Request send_distances_req;
        m_comm.isend(distances, pair_rank, send_distances_req);
        ::MPI_Request recv_distances_req;
        m_comm.irecv(pair_rank, recv_distances, recv_distances_req);

        // Update kNNG on the receiver side
        m_time_recorder.start("Update kNNG receiver");
        num_updated += priv_update_knng(received_checks, true, distances);
        m_time_recorder.stop();

        // Wait for the distances from the pair rank
        m_time_recorder.start("Wait recv dist");
        m_comm.wait(recv_distances_req);
        m_time_recorder.stop();

        // Update kNNG on the sender side
        m_time_recorder.start("Update kNNG sender");
        num_updated += priv_update_knng(assigned_checks, false, recv_distances);
        m_time_recorder.stop();

        m_time_recorder.start("Wait send dist");
        m_comm.wait(send_distances_req);
        m_time_recorder.stop();
      }
    }

    return num_updated;
  }

  void priv_send_fvs(const std::vector<id_pair_type>   &checks,
                     const int                          pair_rank,
                     std::vector<feature_element_type> &fvs_recv,
                     std::vector<std::size_t>          &fv_indices_recv) {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    // Check the limitation of this implementation
    if (std::size_t(point_store.num_points() * num_dims() *
                    sizeof(feature_element_type)) >
        std::size_t(std::numeric_limits<int>::max())) {
      // As MPI uses 'int' for sizes and address offsets, the size of the FV
      // data should be less than 'std::numeric_limits<int>::max()'.
      m_comm.cerr0() << "ERROR: #of local feature vectors is too large."
                     << std::endl;
      m_comm.abort();
    }

    // Find a set of feature vectors to send and the positions of the feature
    // vectors in the packed send buffer.
    std::size_t              num_to_sends_total = 0;
    std::vector<std::size_t> fvs_indices_send;
    if (m_remove_duplicate_fvs) {
      bstuo::unordered_flat_map<id_type, std::size_t> unique_fvs;
      for (const auto &check : checks) {
        const auto &src = check.first;
        if (unique_fvs.count(src) == 0) {
          // The position of the FV is at the end of the packed FV buffer
          const auto idx  = unique_fvs.size();
          unique_fvs[src] = idx;
        }
        fvs_indices_send.push_back(unique_fvs.at(src));
      }
      num_to_sends_total = unique_fvs.size();
      m_profile_db.add("DupFVs", checks.size() - num_to_sends_total);
    } else {
      num_to_sends_total = checks.size();
    }
    m_profile_db.add("SentFVs", num_to_sends_total);

    std::size_t num_to_recvs_total = 0;
    DNND2_CHECK_MPI(::MPI_Sendrecv(
        &num_to_sends_total, 1, mpi::data_type::get<std::size_t>(), pair_rank,
        0, &num_to_recvs_total, 1, mpi::data_type::get<std::size_t>(),
        pair_rank, MPI_ANY_TAG, m_comm.comm(), MPI_STATUS_IGNORE));

    fvs_recv.resize(num_to_recvs_total * num_dims());

    const int num_batches = (std::max(num_to_sends_total, num_to_recvs_total) +
                             m_fv_send_batch_size - 1) /
                            m_fv_send_batch_size;
    m_profile_db.add("FVSendBatches", num_batches);

    std::size_t check_no = 0;
    for (int batch_no = 0; batch_no < num_batches; ++batch_no) {
      const std::size_t num_already_sent =
          std::size_t(batch_no) * m_fv_send_batch_size;
      const auto num_to_send = std::min<std::size_t>(
          m_fv_send_batch_size,
          std::max((ssize_t)num_to_sends_total - (ssize_t)num_already_sent,
                   ssize_t(0)));
      const auto num_to_recv = std::min<std::size_t>(
          m_fv_send_batch_size,
          std::max((ssize_t)num_to_recvs_total - (ssize_t)num_already_sent,
                   ssize_t(0)));

      // Find the addresses of the feature vectors to send
      for (std::size_t n = 0; n < num_to_send; ++check_no) {
        const auto sid = checks.at(check_no).first;
        const auto idx = (fvs_indices_send.size() > 0)
                             ? fvs_indices_send.at(check_no)
                             : check_no;
        if (idx < num_already_sent + n) {
          // This code path is for the case of m_remove_duplicate_fvs is true
          assert(m_remove_duplicate_fvs);
          // This FV was already sent
          continue;
        }
        MPI_Aint base_address;
        MPI_Get_address(point_store.data(), &base_address);
        MPI_Aint fv_address;
        MPI_Get_address(point_store[sid], &fv_address);
        m_fvs_disp[n] = MPI_Aint_diff(fv_address, base_address);
        ++n;
      }

      ::MPI_Datatype custom_fvs_type;
      DNND2_CHECK_MPI(::MPI_Type_create_struct(
          num_to_send, m_fvs_block_lengths.data(), m_fvs_disp.data(),
          m_fvs_types.data(), &custom_fvs_type));
      DNND2_CHECK_MPI(::MPI_Type_commit(&custom_fvs_type));

      DNND2_CHECK_MPI(::MPI_Sendrecv(
          point_store.data(), 1, custom_fvs_type, pair_rank, 0,
          fvs_recv.data() + num_already_sent * num_dims(),
          num_to_recv * num_dims(), mpi::data_type::get<feature_element_type>(),
          pair_rank, MPI_ANY_TAG, m_comm.comm(), MPI_STATUS_IGNORE));
      DNND2_CHECK_MPI(::MPI_Type_free(&custom_fvs_type));
    }

    m_comm.sendrecv_arb_size(pair_rank, fvs_indices_send, fv_indices_recv);
  }

  void priv_cal_distances(const std::vector<id_pair_type>         &checks,
                          const std::vector<feature_element_type> &features,
                          const std::vector<std::size_t>          &indices,
                          std::vector<distance_type> &out_distances) {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    out_distances.resize(checks.size());
    OMP_DIRECTIVE(parallel for)
    for (std::size_t i = 0; i < checks.size(); ++i) {
      const auto pid = checks[i].second;
      assert(priv_owner(pid) == m_comm.rank());
      const auto fv_idx      = (indices.size() > 0) ? indices[i] : i;
      const auto sent_fv_pos = fv_idx * num_dims();
      const auto dist        = m_distance_func(point_store[pid], num_dims(),
                                               &features.at(sent_fv_pos), num_dims());
      out_distances[i]       = dist;
    }
  }

  std::size_t priv_update_knng(const std::vector<id_pair_type>  &checks,
                               const bool                        reverse,
                               const std::vector<distance_type> &distances) {
    std::size_t num_updated = 0;
    OMP_DIRECTIVE(parallel reduction(+ : num_updated)) {
      const auto tid         = omp::get_thread_num();
      const auto num_threads = omp::get_num_threads();
      for (std::size_t i = 0; i < checks.size(); ++i) {
        const auto sid = (reverse) ? checks[i].second : checks[i].first;
        if (sid % (id_type)num_threads != (id_type)tid) {
          continue;
        }
        const auto nid = (reverse) ? checks[i].first : checks[i].second;
        assert(priv_owner(sid) == m_comm.rank());
        num_updated += m_graph.at(sid).push(nid, distances[i],
                                            true);  // push as a new neighbor
      }
    }
    return num_updated;
  }

  std::size_t priv_num_regions() const {
    return priv_num_regions_per_node() * m_comm.num_nodes();
  }

  std::size_t priv_region_no() const { return priv_region_no(m_comm.rank()); }

  std::size_t priv_region_no(const int rank) const {
    return rank / priv_region_size();
  }

  std::size_t priv_num_regions_per_node() const {
    if (m_numa_separate_node) {
      return numa::get_num_avail_nodes();
    } else {
      return 1;
    }
  }

  // The number of ranks in a region
  std::size_t priv_region_size() const {
    if (m_numa_separate_node) {
      if (m_comm.local_size() % priv_num_regions_per_node() != 0) {
        m_comm.cerr0()
            << "ERROR: local_size() % priv_num_regions_per_node() != 0"
            << std::endl;
        m_comm.abort();
      }
      return m_comm.local_size() / priv_num_regions_per_node();
    } else {
      return m_comm.local_size();
    }
  }

  std::size_t priv_regional_rank() const {
    return priv_regional_rank(m_comm.local_rank());
  }

  std::size_t priv_regional_rank(const int rank) const {
    return rank % priv_region_size();
  }

  void priv_cache_popular_fvs(const double popular_fv_ratio) {
    if (popular_fv_ratio == 0.0) {
      m_comm.cout0() << "No popular feature vectors are cached" << std::endl;
      m_min_cache_id = std::numeric_limits<id_type>::max();
      return;
    }

    priv_cout0(m_verbose) << std::endl;
    priv_cout0(true) << "Replicate popular feature vectors" << std::endl;
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    if (popular_fv_ratio >= 1) {
      m_min_cache_id = 0;
    } else {
      m_min_cache_id =
          m_num_total_points - (m_num_total_points * popular_fv_ratio);
    }
    priv_cout0(m_verbose) << "Min ID to replicate: " << m_min_cache_id
                          << std::endl;

    m_time_recorder.start("Replicate popular FVs");
    std::vector<id_type> popular_fvs;
    for (auto itr = point_store.ids_begin(); itr != point_store.ids_end();
         ++itr) {
      if (*itr >= m_min_cache_id) {
        popular_fvs.push_back(*itr);
      }
    }
    {
      const auto max_count    = m_comm.all_reduce_max(popular_fvs.size());
      const auto max_capacity = max_count * priv_num_regions();
      m_pop_fv_store          = std::make_unique<pop_fv_cache_t>(
          num_dims(), max_capacity, priv_region_size(), priv_regional_rank(),
          std::to_string(priv_region_no()), m_comm);
    }

    m_pop_fv_store->create_mine();
    for (std::size_t n = 0; n < m_all_to_all_region_pairs.size(); ++n) {
      const auto pair_cache_no = m_all_to_all_region_pairs[n];
      const auto pair_rank =
          pair_cache_no * priv_region_size() + priv_regional_rank();

      std::vector<id_type> id_recv_buf;
      m_comm.sendrecv_arb_size(pair_rank, popular_fvs, id_recv_buf);

      priv_sendrecv_fvs(
          popular_fvs, pair_rank,
          m_pop_fv_store->my_data() +
              m_pop_fv_store->size(priv_regional_rank()) * num_dims(),
          id_recv_buf.size());

      for (const auto &id : id_recv_buf) {
        assert(priv_owner(id) == pair_rank);
        assert(priv_regional_rank() == priv_regional_rank(pair_rank));
        m_pop_fv_store->register_id(id);
      }
    }
    m_pop_fv_store->finalize();
    m_pop_fv_store->open_as_shared_mode();
    m_time_recorder.stop();
  }

  std::size_t priv_check_cached_neighbors(const int                   pair_rank,
                                          std::vector<id_pair_type> &&checks) {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    if (!m_pop_fv_store) {
      return 0;
    }

    // To receive distances where neighbors' feature vectors are cached
    m_time_recorder.start("Calc dist (cached FVs)");
    std::vector<distance_type> distances(checks.size());
    OMP_DIRECTIVE(parallel for)
    for (std::size_t i = 0; i < checks.size(); ++i) {
      const auto sid = checks[i].first;
      const auto nid = checks[i].second;
      assert(priv_owner(sid) == m_comm.rank());
      const auto *sfv = point_store[sid];
      assert(sfv);
      const auto        nfv_bank_no = priv_regional_rank(priv_owner(nid));
      const auto *const nfv         = m_pop_fv_store->get(nfv_bank_no, nid);
      assert(nfv);
      const auto dist = m_distance_func(sfv, num_dims(), nfv, num_dims());
      distances[i]    = dist;
    }
    m_time_recorder.stop();

    MPI_Request send_checks_req;
    m_comm.isend(checks.data(), checks.size(), m_id_pair_type, pair_rank,
                 send_checks_req);

    MPI_Request send_distances_req;
    m_comm.isend(distances, pair_rank, send_distances_req);

    std::vector<id_pair_type> recv_checks;
    MPI_Request               recv_checks_req;
    m_comm.irecv(pair_rank, recv_checks, m_id_pair_type, recv_checks_req);

    std::vector<distance_type> recv_distances;
    MPI_Request                recv_distances_req;
    m_comm.irecv(pair_rank, recv_distances, recv_distances_req);

    m_time_recorder.start("Update kNNG sender (cached FVs)");
    std::size_t num_updated = priv_update_knng(checks, false, distances);
    m_time_recorder.stop();

    m_time_recorder.start("Wait for CHKs&DISTs (cached FVs)");
    m_comm.wait(recv_checks_req);
    m_comm.wait(recv_distances_req);
    m_time_recorder.stop();

    m_time_recorder.start("Update kNNG receiver (cached FVs)");
    num_updated += priv_update_knng(recv_checks, true, recv_distances);
    m_time_recorder.stop();

    m_comm.wait(send_checks_req);
    m_comm.wait(send_distances_req);

    return num_updated;
  }

  void priv_sendrecv_fvs(const std::vector<id_type> &ids_to_send,
                         const int pair_rank, feature_element_type *recv_buf,
                         const std::size_t total_recv_count) {
    const auto &point_store = *(m_point_stores.at(priv_regional_rank()));

    // Check the limitation of this implementation
    if (std::size_t(point_store.num_points() * num_dims() *
                    sizeof(feature_element_type)) >
        std::size_t(std::numeric_limits<int>::max())) {
      // As MPI uses 'int' for sizes and address offsets, the size of the FV
      // data should be less than 'std::numeric_limits<int>::max()'.
      m_comm.cerr0() << "ERROR: #of local feature vectors is too large."
                     << std::endl;
      m_comm.abort();
    }

    const std::size_t batch_size = m_fv_send_batch_size;
    const std::size_t num_batches =
        (std::max(ids_to_send.size(), total_recv_count) + batch_size - 1) /
        batch_size;

    std::size_t num_sent   = 0;
    std::size_t num_recved = 0;
    for (int i = 0; i < num_batches; ++i) {
      const std::size_t num_to_send =
          (i * batch_size >= ids_to_send.size())
              ? 0
              : std::min(batch_size, (ids_to_send.size() - i * batch_size));
      const std::size_t num_to_recv =
          (i * batch_size >= total_recv_count)
              ? 0
              : std::min(batch_size, (total_recv_count - i * batch_size));

      // Create a custom data type for the feature vectors to send
      // This technique allows us to avoid creating a packed FV send buffer on
      // our side.
      for (std::size_t n = 0; n < num_to_send; ++n) {
        const auto idx = i * batch_size + n;
        MPI_Aint   base_address;
        MPI_Get_address(point_store.data(), &base_address);
        MPI_Aint fv_address;
        MPI_Get_address(point_store[ids_to_send.at(idx)], &fv_address);
        m_fvs_disp.at(n) = MPI_Aint_diff(fv_address, base_address);
      }
      ::MPI_Datatype custom_fvs_type;
      DNND2_CHECK_MPI(::MPI_Type_create_struct(
          num_to_send, m_fvs_block_lengths.data(), m_fvs_disp.data(),
          m_fvs_types.data(), &custom_fvs_type));
      DNND2_CHECK_MPI(::MPI_Type_commit(&custom_fvs_type));

      auto *recv_buf_current_head = recv_buf + num_recved * num_dims();
      DNND2_CHECK_MPI(
          ::MPI_Sendrecv(point_store.data(), 1, custom_fvs_type, pair_rank, 0,
                         recv_buf_current_head, num_to_recv * num_dims(),
                         mpi::data_type::get<feature_element_type>(), pair_rank,
                         MPI_ANY_TAG, m_comm.comm(), MPI_STATUS_IGNORE));

      num_sent += num_to_send;
      num_recved += num_to_recv;
      DNND2_CHECK_MPI(::MPI_Type_free(&custom_fvs_type));
    }
    assert(num_sent == ids_to_send.size());
    assert(num_recved == total_recv_count);
  }

  distance_function  &m_distance_func;
  time_recorder_base &m_time_recorder;
  mpi::communicator  &m_comm;
  std::mt19937_64 m_rng;  // Must be initialized after m_comm as it uses rank
  bool            m_verbose{false};
  bool            m_remove_duplicate_fvs{false};
  knng_heap_t     m_graph{};
  std::size_t     m_k{0};
  double          m_rho{1.0};
  double          m_delta{0.001};
  std::size_t     m_num_total_points{0};
  profile_db      m_profile_db;
  // For sending feature vectors
  int                         m_fv_send_batch_size{0};
  ::MPI_Datatype              m_nb_dist_type{MPI_DATATYPE_NULL};
  ::MPI_Datatype              m_id_pair_type{MPI_DATATYPE_NULL};
  ::MPI_Datatype              m_feature_type{MPI_DATATYPE_NULL};
  std::vector<::MPI_Aint>     m_fvs_disp{};
  std::vector<int>            m_fvs_block_lengths{};
  std::vector<::MPI_Datatype> m_fvs_types{};
  std::vector<int>            m_all_to_all_pairs{};
  std::vector<int>            m_all_to_all_region_pairs{};
#ifdef PROFILE_FV
  // Counts how many times each feature vector was received.
  bstuo::unordered_flat_map<id_type, std::size_t> m_fv_count;
#endif
  int                              m_super_step_no{0};
  std::unique_ptr<pop_fv_cache_t>  m_pop_fv_store;
  std::size_t                      m_min_cache_id{0};
  bool                             m_share_pstore_regionally{false};
  std::vector<const point_store *> m_point_stores;
  std::vector<metall::manager *>   m_point_store_managers;
  std::size_t                      m_num_dims{0};
  bool                             m_numa_separate_node{false};
};
}  // namespace saltatlas::neo_dnnd

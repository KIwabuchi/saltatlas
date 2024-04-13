// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/unordered_map.hpp>
#include <metall/metall.hpp>
#include <perroht/unordered_map.hpp>

#include "saltatlas/neo_dnnd/hash.hpp"
#include "saltatlas/neo_dnnd/mpi.hpp"
#include "saltatlas/neo_dnnd/numa.hpp"

namespace saltatlas::neo_dnnd::ndndtl {

namespace {
namespace bip = boost::interprocess;
}

// TODO: change API
/// \brief Interprocess shared feature vector pool.
/// There are multiple banks, each bank can contain the same number of feature
/// vectors.
template <typename id_type, typename feature_element_type>
class pfv_replica_store {
 private:
  using index_type       = uint32_t;
  using seg_allocator    = metall::manager::allocator_type<std::byte>;
  using index_table_type = perroht::unordered_flat_map<
      id_type, index_type, saltatlas::neo_dnnd::hash<1314>, std::equal_to<>,
      typename std::allocator_traits<seg_allocator>::template rebind_alloc<
          std::pair<const id_type, index_type>>>;

 public:
  static std::size_t get_element_size(const size_t dims) {
    std::size_t sz = dims * sizeof(feature_element_type);
    sz += sizeof(typename index_table_type::value_type) + 1;
    return sz;
  }

  pfv_replica_store(const size_t dims, const size_t single_capacity,
                    const size_t num_banks, const size_t my_bank_no,
                    const std::string &cache_name, mpi::communicator &comm)
      : m_dims(dims),
        m_single_capacity(single_capacity),
        m_num_banks(num_banks),
        m_my_bank_no(my_bank_no),
        m_cache_name(cache_name),
        m_comm(comm) {
    m_segment_managers.resize(m_num_banks, nullptr);
    m_fv_pools.resize(m_num_banks, nullptr);
    m_index_tables.resize(m_num_banks, nullptr);
  }

  ~pfv_replica_store() {
    m_fv_pools.clear();
    m_index_tables.clear();
    for (auto *manager : m_segment_managers) {
      delete manager;
    }
    m_comm.local_barrier();
  }

  void create_mine() {
    m_segment_managers[m_my_bank_no] = new metall::manager(
        metall::create_only, priv_gen_shm_name(m_my_bank_no));
    auto *manager = m_segment_managers[m_my_bank_no];

    m_fv_pools[m_my_bank_no] =
        static_cast<feature_element_type *>(manager->allocate(
            m_dims * m_single_capacity * sizeof(feature_element_type)));
    manager->construct<metall::offset_ptr<feature_element_type>>(
        metall::unique_instance)(m_fv_pools[m_my_bank_no]);

    m_index_tables[m_my_bank_no] = manager->construct<index_table_type>(
        metall::unique_instance)(manager->template get_allocator<std::byte>());
  }

  void finalize() {
    delete m_segment_managers[m_my_bank_no];
    m_segment_managers[m_my_bank_no] = nullptr;
    m_comm.local_barrier();
  }

  void open_as_shared_mode() {
    for (int i = 0; i < m_num_banks; ++i) {
      m_segment_managers[i] =
          new metall::manager(metall::open_read_only, priv_gen_shm_name(i));
      auto *pool_offset_ptr =
          m_segment_managers[i]
              ->find<metall::offset_ptr<feature_element_type>>(
                  metall::unique_instance)
              .first;
      m_fv_pools[i] = metall::to_raw_pointer(*pool_offset_ptr);
      assert(m_fv_pools[i]);
      m_index_tables[i] = m_segment_managers[i]
                              ->find<index_table_type>(metall::unique_instance)
                              .first;
      assert(m_index_tables[i]);
    }
    m_comm.local_barrier();
  }

  bool register_id(const id_type id) {
    const auto bank_no = m_my_bank_no;
    if (m_index_tables.at(bank_no)->count(id) > 0) {
      return false;
    }
    const auto index                    = m_index_tables.at(bank_no)->size();
    (*(m_index_tables.at(bank_no)))[id] = index;
    return true;
  }

  const feature_element_type *get(const int bank_no, const id_type id) const {
    assert(bank_no < m_num_banks);
    assert(bank_no < m_index_tables.size());
    assert(m_index_tables.at(bank_no)->count(id) > 0);
    const auto idx = m_index_tables.at(bank_no)->at(id);
    return m_fv_pools.at(bank_no) + idx * m_dims;
  }

  feature_element_type *my_data() const { return m_fv_pools.at(m_my_bank_no); }

  std::size_t size(const int bank_no) const {
    return m_index_tables.at(bank_no)->size();
  }

 private:
  std::string priv_gen_shm_name(const int id) const {
    std::filesystem::path path;
#ifdef __APPLE__
    path = "/tmp/";
#else
    path = "/dev/shm/";
#endif
    path /= "neo_dnnd-pcache" + m_cache_name + "-" + std::to_string(id);
    return path.string();
  }

  const size_t                        m_dims;
  const size_t                        m_single_capacity;
  const size_t                        m_num_banks;
  const size_t                        m_my_bank_no;
  const std::string                   m_cache_name;
  mpi::communicator                  &m_comm;
  std::vector<feature_element_type *> m_fv_pools;
  std::vector<index_table_type *>     m_index_tables;
  std::vector<metall::manager *>      m_segment_managers;
};

}  // namespace saltatlas::neo_dnnd::ndndtl
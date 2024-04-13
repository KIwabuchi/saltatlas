// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#define USE_COMPACT_MAP_IN_KNN_LIST

#include <cassert>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <vector>
#include <queue>

#include <boost/container/flat_map.hpp>

#include "saltatlas/neo_dnnd/memory.hpp"
#include "saltatlas/neo_dnnd/float.hpp"

namespace saltatlas::neo_dnnd::ndndtl {

template <typename Id, typename Distance>
struct neighbor {
  using id_type = Id;
  using distance_type = Distance;

  neighbor() = default;

  neighbor(const id_type& _id, const distance_type& _distance)
      : id(_id), distance(_distance) {}

  friend bool operator<(const neighbor& lhd, const neighbor& rhd) {
    if (lhd.distance != rhd.distance) return lhd.distance < rhd.distance;
    return lhd.id < rhd.id;
  }

  template <typename T1, typename T2>
  friend std::ostream& operator<<(std::ostream& os, const neighbor<T1, T2>& ng);

  id_type id;
  distance_type distance;
};

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const neighbor<T1, T2>& ng) {
  os << "id = " << ng.id << ", distance = " << ng.distance;
  return os;
}

template <typename Id, typename Distance>
inline bool operator==(const neighbor<Id, Distance>& lhs,
                       const neighbor<Id, Distance>& rhs) {
  return lhs.id == rhs.id && nearly_equal(lhs.distance, rhs.distance);
}

template <typename Id, typename Distance>
inline bool operator!=(const neighbor<Id, Distance>& lhs,
                       const neighbor<Id, Distance>& rhs) {
  return !(lhs == rhs);
}

template <typename Key, typename T>
class compact_map {
 public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<Key, T>;
  using const_iterator = const value_type*;

  compact_map() = default;

  compact_map(const compact_map&) = delete;
  compact_map& operator=(const compact_map&) = delete;
  compact_map(compact_map&&) = default;
  compact_map& operator=(compact_map&&) = default;

  ~compact_map() = default;

  void reserve(const std::size_t capacity) {
    m_data = std::make_unique<value_type[]>(capacity);
    if (m_data == nullptr) {
      std::cerr << "Failed to allocate memory for compact_map" << std::endl;
      std::abort();
    }

    if (capacity >= std::numeric_limits<uint16_t>::max()) {
      std::cerr << "The capacity of compact_map is too large" << std::endl;
      std::abort();
    }
    m_capacity = capacity;
  }

  std::size_t capacity() const { return m_capacity; }

  std::size_t size() const { return m_size; }

  std::size_t count(const key_type& key) const {
    for (std::size_t i = 0; i < m_size; ++i) {
      if (m_data[i].first == key) {
        return 1;
      }
    }
    return 0;
  }

  const mapped_type& at(const key_type& key) const {
    for (std::size_t i = 0; i < m_size; ++i) {
      if (m_data[i].first == key) {
        return m_data[i].second;
      }
    }
    std::cerr << "Key not found in compact_map" << std::endl;
    std::abort();
  }

  mapped_type& at(const key_type& key) {
    return const_cast<mapped_type&>(
        static_cast<const compact_map*>(this)->at(key));
  }

  const_iterator begin() const { return m_data.get(); }
  const_iterator end() const { return m_data.get() + m_size; }

  void emplace(key_type key, mapped_type value) {
    if (m_size == m_capacity) {
      std::cerr << "compact_map is full" << std::endl;
      std::abort();
    }
    m_data[m_size++] = std::make_pair(std::move(key), std::move(value));
  }

  void erase(const key_type& key) {
    for (std::size_t i = 0; i < m_size; ++i) {
      if (m_data[i].first == key) {
        if (m_capacity == 1) {
          m_size = 0;
        } else {
          m_data[i] = m_data[m_size - 1];
          --m_size;
        }
        return;
      }
    }
  }

 private:
  std::unique_ptr<value_type[]> m_data;
  uint16_t m_capacity{0};
  uint16_t m_size{0};
  ;
};

/// \brief A heap that keeps the k-nearest neighbors, avoiding duplicate
/// neighbors.
// TODO: make a version that deos not take value?
template <typename Id, typename Distance, typename Value = std::byte>
class knn_list {
 public:
  using id_type = Id;
  using distance_type = Distance;
  using value_type = Value;
  using neighbor_type = neighbor<id_type, distance_type>;

 private:
  using heap_type = std::priority_queue<neighbor_type>;

#ifdef USE_COMPACT_MAP_IN_KNN_LIST
  using map_type = compact_map<id_type, value_type>;
#else
  using map_type = boost::container::flat_map<id_type, value_type>;
#endif

 public:
  knn_list() = default;

  void init(std::size_t k) {
#ifndef USE_COMPACT_MAP_IN_KNN_LIST
    if (k > std::numeric_limits<uint16_t>::max()) {
      std::cerr << "The capacity of knn_list is too large" << std::endl;
      std::abort();
    }
    m_k = k;
#endif
    // m_knn_heap.reserve(k);
    m_map.reserve(k);
  }

  /// \brief Push a neighbor if it is closer than the current neighbors and is
  /// not one of the current neighbors.
  /// \param id Neighbor ID.
  /// \param d Distance.
  /// \param v Value associated with the neighbor.
  /// \return True if the neighbor has been pushed; otherwise, false.
  bool push(const id_type& id, const distance_type& d,
            value_type v = value_type{}) {
    if (contains(id)) {
      return false;
    }

    if (m_knn_heap.size() < k()) {
      priv_push_nocheck(id, d, std::move(v));
      return true;
    }

    if (m_knn_heap.top().distance > d) {
      pop();
      priv_push_nocheck(id, d, std::move(v));
      return true;
    }

    return false;
  }

  const neighbor_type& top() const { return m_knn_heap.top(); }

  void pop() {
    assert(m_map.count(m_knn_heap.top().id) > 0);
    m_map.erase(m_knn_heap.top().id);
    m_knn_heap.pop();
  }

  bool contains(const id_type& id) const { return m_map.count(id); }

  // Provide only const iterators to prevent the user from modifying the IDs.
  typename map_type::const_iterator begin() const { return m_map.begin(); }

  // Provide only const iterators to prevent the user from modifying the IDs.
  typename map_type::const_iterator end() const { return m_map.end(); }

  value_type& value(const id_type& id) { return m_map.at(id); }

  const value_type& value(const id_type& id) const { return m_map.at(id); }

  std::size_t size() const { return m_knn_heap.size(); }

  bool empty() const { return m_knn_heap.empty(); }

  std::size_t k() const { return m_map.capacity(); }

  /// \brief Get neighbors.
  std::vector<neighbor_type> extract_neighbors() const {
    std::vector<neighbor_type> neighbors;
    neighbors.reserve(m_knn_heap.size());
    auto tmp = m_knn_heap;
    while (!tmp.empty()) {
      neighbors.emplace_back(tmp.top());
      tmp.pop();
    }

    std::reverse(neighbors.begin(), neighbors.end());
    return neighbors;
  }

 private:
  void priv_push_nocheck(const id_type& id, const distance_type& d,
                         value_type v) {
    m_knn_heap.emplace(id, d);
    m_map.emplace(id, std::move(v));
  }

  heap_type m_knn_heap{};
  map_type m_map{};
#ifndef USE_COMPACT_MAP_IN_KNN_LIST
  uint16_t m_k{0};
#endif
};

}  // namespace saltatlas::neo_dnnd::ndndtl
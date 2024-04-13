// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <cassert>
#include <perroht/unordered_map.hpp>

#include "saltatlas/neo_dnnd/hash.hpp"
#include "saltatlas/neo_dnnd/memory.hpp"

namespace saltatlas::neo_dnnd {

/// \brief Dataset store. There are N points in the dataset, each point is
/// represented by a feature vector. Every point has the same dimension.
template <typename _id_type, typename _value_type,
          typename _allocator_type = std::allocator<_value_type>>
class point_store {
 public:
  using id_type = _id_type;
  using value_type = _value_type;
  using allocator_type = typename std::allocator_traits<
      _allocator_type>::template rebind_alloc<value_type>;

 private:
  using pointer = typename std::pointer_traits<
      typename allocator_type::pointer>::template rebind<value_type>;

  using id_table_t = perroht::unordered_flat_map<
      id_type, id_type, saltatlas::neo_dnnd::hash<13149>, std::equal_to<>,
      typename std::allocator_traits<allocator_type>::template rebind_alloc<
          std::pair<const id_type, id_type>>>;

 public:
  class const_id_iterator;

  /// \brief Constructor.
  explicit point_store(const allocator_type &allocator = allocator_type{})
      : m_id_map(allocator), m_allocator(allocator) {}

  /// \brief Destructor.
  ~point_store() {
    if (m_data) {
      m_allocator.deallocate(m_data, m_num_points * m_num_dims);
    }
  }

  inline value_type *operator[](id_type id) {
    if (m_id_map.count(id) == 0) {
      const id_type internal_id = m_id_map.size();
      m_id_map[id] = internal_id;
    }
    assert(m_id_map.count(id) > 0);
    const id_type internal_id = m_id_map.at(id);
    assert(internal_id < m_num_points);
    return data() + internal_id * m_num_dims;
  }

  inline value_type *operator[](id_type id) const {
    assert(m_id_map.count(id) > 0);
    const id_type internal_id = m_id_map.at(id);
    assert(internal_id < m_num_points);
    return data() + internal_id * m_num_dims;
  }

  bool contains(id_type id) const { return m_id_map.count(id) > 0; }

  inline value_type *data() const { return metall::to_raw_pointer(m_data); }

  inline std::size_t num_points() const { return m_id_map.size(); }

  inline std::size_t dim() const { return m_num_dims; }

  bool init(std::size_t num_points, std::size_t num_dims) {
    m_num_points = num_points;
    m_num_dims = num_dims;

    m_id_map.reserve(m_num_points);

    assert(!m_data);
    m_data = m_allocator.allocate(m_num_points * m_num_dims);
    return !!m_data;
  }

  const_id_iterator ids_begin() const {
    return const_id_iterator(m_id_map.cbegin());
  }

  const_id_iterator ids_begin() { return const_id_iterator(m_id_map.cbegin()); }

  const_id_iterator ids_end() const {
    return const_id_iterator(m_id_map.cend());
  }

  const_id_iterator ids_end() { return const_id_iterator(m_id_map.cend()); }

 private:
  std::size_t m_num_points{0};
  std::size_t m_num_dims{0};
  pointer m_data{nullptr};
  id_table_t m_id_map;
  utl::other_allocator<allocator_type, value_type> m_allocator;
};

template <typename _id_type, typename _value_type, typename _allocator_type>
class point_store<_id_type, _value_type, _allocator_type>::const_id_iterator {
 private:
  using internal_iterator = typename id_table_t::const_iterator;

 public:
  using id_type = _id_type;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = const id_type *;
  using reference = const id_type &;

  const_id_iterator() = default;

  explicit const_id_iterator(internal_iterator itr) : m_it(itr) {}

  const_id_iterator &operator++() {
    ++m_it;
    return *this;
  }

  const_id_iterator operator++(int) {
    const_id_iterator tmp(*this);
    ++m_it;
    return tmp;
  }

  bool operator==(const const_id_iterator &rhs) const {
    return m_it == rhs.m_it;
  }

  bool operator!=(const const_id_iterator &rhs) const {
    return m_it != rhs.m_it;
  }

  reference operator*() const { return m_it->first; }

  pointer operator->() const { return &m_it->first; }

 private:
  internal_iterator m_it;
};

}  // namespace saltatlas::neo_dnnd
// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string_view>
#include <string>

namespace saltatlas::neo_dnnd::system {

/// \brief Reads a value from /proc/meminfo
/// \param key Target token looking for
/// \return On success, returns read value. On error, returns -1.
inline ssize_t read_meminfo(const std::string_view &key) {
  std::ifstream fin("/proc/meminfo");
  if (!fin.is_open()) {
    return -1;
  }

  std::string key_with_colon(key);
  if (key_with_colon.at(key_with_colon.length() - 1) != ':') {
    key_with_colon.append(":");
  }
  std::string token;
  while (fin >> token) {
    if (token != key_with_colon) continue;

    ssize_t value;
    if (!(fin >> value)) {
      return -1;
    }

    std::string unit;
    if (fin >> unit) {
      std::transform(unit.begin(), unit.end(), unit.begin(),
                     [](const unsigned char c) { return std::tolower(c); });
      if (unit == "kb") {  // for now, we only expect this case
        return value * 1024;
      }
      return -1;
    } else {  // found a line does not have a unit
      return value;
    }
  }
  return -1;
}

/// \brief Returns the size of the total ram size
/// \return On success, returns the total ram size of the system. On error,
/// returns -1.
inline ssize_t get_total_ram_size() {
  const ssize_t mem_total = read_meminfo("MemTotal:");
  if (mem_total == -1) {
    return -1;  // something wrong;
  }
  return static_cast<ssize_t>(mem_total);
}

/// \brief Returns the size of used ram size from /proc/meminfo
/// \return On success, returns the used ram size. On error, returns -1.
inline ssize_t get_used_ram_size() {
  const ssize_t mem_total = read_meminfo("MemTotal:");
  const ssize_t mem_free = read_meminfo("MemFree:");
  const ssize_t buffers = read_meminfo("Buffers:");
  const ssize_t cached = read_meminfo("Cached:");
  const ssize_t slab = read_meminfo("Slab:");
  const ssize_t used = mem_total - mem_free - buffers - cached - slab;
  if (mem_total == -1 || mem_free == -1 || buffers == -1 || cached == -1 ||
      slab == -1 || used < 0) {
    return -1;  // something wrong;
  }
  return used;
}

/// \brief Returns the size of free ram size from /proc/meminfo
/// \return On success, returns the free ram size. On error, returns -1.
inline ssize_t get_free_ram_size() {
  const ssize_t mem_free = read_meminfo("MemFree:");
  if (mem_free == -1) {
    return -1;  // something wrong;
  }
  return static_cast<ssize_t>(mem_free);
}

/// \brief Returns the size of the 'cached' ram size
/// \return On success, returns the 'cached' ram size of the system. On error,
/// returns -1.
inline ssize_t get_page_cache_size() {
  const ssize_t cached_size = read_meminfo("Cached:");
  if (cached_size == -1) {
    return -1;  // something wrong;
  }
  return static_cast<ssize_t>(cached_size);
}

/// \brief Returns the size of the 'MemAvailable' ram size from /proc/meminfo
/// \return On success, returns the 'MemAvailable' ram size. On error, returns
/// -1.
inline ssize_t get_available_ram_size() {
  const ssize_t mem_available = read_meminfo("MemAvailable:");
  if (mem_available == -1) {
    return -1;  // something wrong;
  }
  return static_cast<ssize_t>(mem_available);
}

}  // namespace saltatlas::neo_dnnd::system
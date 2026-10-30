// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "saltatlas/solanet/detail/cuvs_nn/utils.hpp"

namespace saltatlas::solanet::cuvs_nn {
// Basic utility functions for HIP/ROCM backend
template <typename T>
T* allocate_cuda_host_memory(const size_t n_elements) {
  // Allocate memory on CUDA device
  T* d_ptr = nullptr;

  spdlog::trace("Allocating CUDA memory: {} GB", n_elements * sizeof(T) / 1e9);
  SOLANET_CUDA_CHECK(
      cudaHostAlloc(&d_ptr, n_elements * sizeof(T), cudaHostAllocDefault));
  spdlog::trace("CUDA memory allocated at address: {}", (void*)d_ptr);
  return d_ptr;
}

template <typename T>
void free_cuda_host_memory(T* d_ptr) {
  // Free memory on CUDA device
  if (d_ptr != nullptr) {
    spdlog::trace("Freeing CUDA memory at address: {}", (void*)d_ptr);
    SOLANET_CUDA_CHECK(cudaFreeHost(d_ptr));
  }
}

template <typename T>
struct cuda_host_mem_deleter {
  void operator()(T* d_ptr) const noexcept { free_cuda_host_memory(d_ptr); }
};

template <typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, cuda_host_mem_deleter<T>>;

template <typename T>
cuda_host_unique_ptr<T> make_cuda_host_array(const size_t n_elements) {
  return cuda_host_unique_ptr<T>(allocate_cuda_host_memory<T>(n_elements));
}

// STL-compatible allocator that uses cudaHostAlloc internally
template <typename T>
class cuda_host_allocator {
 public:
  using value_type      = T;
  cuda_host_allocator() = default;
  template <typename U>
  cuda_host_allocator(const cuda_host_allocator<U>&) noexcept {}

  T* allocate(const size_t n) {
    void*      ptr = nullptr;
    const auto err = cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t /*n*/) noexcept {
    if (p) {
      cudaFreeHost(p);
    }
  }
};

// Get aligned array from 'already allocated' shared memory.
// Advances the pointer accordingly
// Returns the allocated pointer
// Assumes that there is enough space in shared memory.
// No bounds checking is performed.
template <typename T>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE T* get_shared_mem(
    unsigned char*& ptr, const int count, const int n_warps_per_block,
    const int local_warp_id) {
  const uintptr_t addr         = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t aligned_addr = (addr + alignof(T) - 1) & ~(alignof(T) - 1);
  ptr                          = reinterpret_cast<unsigned char*>(aligned_addr);
  T* const typed_ptr = reinterpret_cast<T*>(ptr) + local_warp_id * count;
  ptr += sizeof(T) * count * n_warps_per_block;
  return typed_ptr;
};

}  // namespace saltatlas::solanet::cuvs_nn

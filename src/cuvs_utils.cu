#include "saltatlas/solanet/detail/cuvs_nn/utils.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace saltatlas::solanet::gpu {

__global__ void update_knng_kernel(
    uint32_t* all_master_nids, float* all_master_dists,
    const uint32_t* all_cad_nids, const float* all_cad_dists,
    const int n_points, const int master_degree, const int candidate_degree,
    const uint32_t candidate_id_offset, const uint32_t master_id_offset,
    unsigned long long* num_updates, uint32_t* all_work_nids,
    float* all_work_dists) {
  // Use partial merge sort to update knng for each point
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;

  // Get the current point's knng
  const uint32_t* cad_nids  = all_cad_nids + idx * candidate_degree;
  const float*    cad_dists = all_cad_dists + idx * candidate_degree;

  // Get the master knng
  uint32_t* master_nids  = all_master_nids + idx * master_degree;
  float*    master_dists = all_master_dists + idx * master_degree;

  // Use the passed work arrays
  uint32_t* work_nids  = all_work_nids + idx * master_degree;
  float*    work_dists = all_work_dists + idx * master_degree;

  int                ni      = 0;
  int                mi      = 0;
  int                wi      = 0;
  unsigned long long updates = 0;
  while (wi < master_degree && (ni < candidate_degree || mi < master_degree)) {
    const bool take_candidate =
        mi >= master_degree ||
        (ni < candidate_degree && cad_dists[ni] < master_dists[mi]);
    const uint32_t nid  = take_candidate ? cad_nids[ni] + candidate_id_offset
                                         : master_nids[mi] + master_id_offset;
    const float    dist = take_candidate ? cad_dists[ni] : master_dists[mi];
    if (take_candidate) {
      ++ni;
    } else {
      ++mi;
    }

    bool duplicate = false;
    for (int i = 0; i < wi; ++i) {
      if (work_nids[i] == nid) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;

    work_nids[wi]  = nid;
    work_dists[wi] = dist;
    ++wi;
    updates += take_candidate;
  }

  // Copy the work space back to the master knng
  for (int wi = 0; wi < master_degree; wi++) {
    master_nids[wi]  = work_nids[wi];
    master_dists[wi] = work_dists[wi];
  }

  // Add updates to the global counter
  atomicAdd(num_updates, updates);
}

template <typename id_type, typename dist_type>
size_t update_knng(id_type* master_nids, dist_type* master_dists,
                   const id_type*   candidate_nids,
                   const dist_type* candidate_dists, size_t n_points,
                   int master_degree, int candidate_degree,
                   id_type candidate_id_offset, id_type master_id_offset,
                   raft::device_resources& dev_resources) {
  if (n_points == 0) return 0;
  assert(master_degree > 0);
  assert(candidate_degree > 0);

  constexpr size_t batch_mem_size = 1ULL << 27;
  const size_t     per_point_mem_size =
      size_t(master_degree) * (sizeof(id_type) + sizeof(dist_type));
  size_t batch_size = (batch_mem_size / per_point_mem_size);
  // if (batch_size > std::numeric_limits<uint32_t>::max()) {
  //   throw std::runtime_error("Batch size is too large");
  // }

  // Make batch_size to be a multiple of 32
  batch_size = (batch_size / 32) * 32;

  auto d_n_update =
      raft::make_device_scalar<unsigned long long>(dev_resources, 0);
  auto d_work_nids = raft::make_device_matrix<id_type, uint64_t>(
      dev_resources, batch_size, master_degree);
  auto d_work_dists = raft::make_device_matrix<dist_type, uint64_t>(
      dev_resources, batch_size, master_degree);

  unsigned long long h_n_updates = 0;
  auto stream = raft::resource::get_cuda_stream(dev_resources);

  for (size_t count = 0; count < n_points; count += batch_size) {
    size_t current_batch_size = std::min(batch_size, n_points - count);
    size_t master_offset      = count * master_degree;
    size_t candidate_offset   = count * candidate_degree;

    const int threadsPerBlock = 256;
    const int numBlocks =
        (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;
    update_knng_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        master_nids + master_offset, master_dists + master_offset,
        candidate_nids + candidate_offset, candidate_dists + candidate_offset,
        current_batch_size, master_degree, candidate_degree,
        candidate_id_offset, master_id_offset, d_n_update.data_handle(),
        d_work_nids.data_handle(), d_work_dists.data_handle());

    unsigned long long tmp    = 0;
    raft::update_host(
        &tmp, const_cast<const unsigned long long*>(d_n_update.data_handle()),
        1, stream);
    raft::resource::sync_stream(dev_resources, stream);
    // The device counter is cumulative across batches.
    h_n_updates = tmp;
  }

  return h_n_updates;
}

template size_t update_knng<uint32_t, float>(
    uint32_t* master_nids, float* master_dists, const uint32_t* candidate_nids,
    const float* candidate_dists, size_t n_points, int master_degree,
    int candidate_degree, uint32_t candidate_id_offset,
    uint32_t master_id_offset, raft::device_resources& dev_resources);

template <typename T>
void add_scalar(T* array, T scalar, size_t n) {
  // Use thrust to add a scalar to an array
  thrust::device_ptr<T> array_ptr(array);
  thrust::transform(array_ptr, array_ptr + n, array_ptr,
                    thrust::placeholders::_1 + scalar);
}

template void add_scalar<float>(float* array, float scalar, size_t n);
template void add_scalar<uint32_t>(uint32_t* array, uint32_t scalar, size_t n);

void cast_search_ids(uint32_t* dst, const int64_t* src, size_t length,
                     raft::device_resources& dev_resources) {
  const auto stream = raft::resource::get_cuda_stream(dev_resources);
  thrust::transform(thrust::cuda::par.on(stream), src, src + length, dst,
                    [] __device__(const int64_t id) {
                      return static_cast<uint32_t>(id);
                    });
}

}  // namespace saltatlas::solanet::gpu

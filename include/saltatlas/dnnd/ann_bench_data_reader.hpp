#pragma once

#include <string_view>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/utilities/general.hpp>
#include <saltatlas/dnnd/detail/utilities/string_cast.hpp>

namespace saltatlas::ann_bench_data_reader {

template <typename id_t, typename T, typename pstore_alloc>
inline void read_points(
    std::string_view                              file_name,
    dndetail::point_store<id_t, T, pstore_alloc> &local_point_store,
    const std::function<int(const id_t &id)>     &point_partitioner,
    ygm::comm &comm, const bool verbose) {
  std::ifstream ifs(file_name.data(), std::ios::binary);
  if (!ifs.is_open()) {
    comm.cerr0() << "Failed to open " << file_name << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  uint32_t num_points     = 0;
  uint32_t num_dimensions = 0;
  if (!ifs.read(reinterpret_cast<char *>(&num_points), sizeof(num_points)) ||
      !ifs.read(reinterpret_cast<char *>(&num_dimensions),
                sizeof(num_dimensions))) {
    comm.cerr() << "Failed to read from " << file_name << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  comm.cf_barrier();

  if (verbose) {
    comm.cout0() << "#of point\t" << num_points << std::endl;
    comm.cout0() << "#of dimension\t" << num_dimensions << std::endl;
  }

  const auto feature_size = num_dimensions * sizeof(T);
  const auto range =
      dndetail::partial_range(num_points, comm.rank(), comm.size());
  ifs.seekg(range.first * feature_size);
  const auto     num_reads = range.second - range.first;
  std::vector<T> feature(feature_size);
  ygm::ygm_ptr<dndetail::point_store<id_t, T, pstore_alloc>> ptr_point_store(
      &local_point_store);
  for (std::size_t i = 0; i < num_reads; ++i) {
    if (!ifs.read(reinterpret_cast<char *>(feature.data()), feature_size)) {
      comm.cerr() << "Failed to read from " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const id_t id = range.first + i;

    // Send to the corresponding process
    comm.async(
        point_partitioner(id),
        [](auto, const id_t id, const auto &sent_feature,
           auto ptr_point_store) {
          auto &feature = ptr_point_store->feature_vector(id);
          feature.insert(feature.begin(), sent_feature.begin(),
                         sent_feature.end());
        },
        id, feature, ptr_point_store);
  }
  comm.barrier();
}

template <typename dnnd_type>
inline void read_queries(
    std::string_view                            query_file,
    typename dnnd_type::query_point_store_type &query_points,
    const bool                                  verbose) {
  using id_t = typename dnnd_type::id_type;
  using T    = typename dnnd_type::feature_element_type;

  std::ifstream ifs(query_file.data(), std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << query_file << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  uint32_t num_queries    = 0;
  uint32_t num_dimensions = 0;
  if (!ifs.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries)) ||
      !ifs.read(reinterpret_cast<char *>(&num_dimensions),
                sizeof(num_dimensions))) {
    std::cerr << "Failed to read from " << query_file << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (verbose) {
    std::cout << "#of point\t" << num_queries << std::endl;
    std::cout << "#of dimension\t" << num_dimensions << std::endl;
  }

  const auto     feature_size = num_dimensions * sizeof(T);
  std::vector<T> feature(feature_size);
  for (id_t id = 0; id < num_queries; ++id) {
    if (!ifs.read(reinterpret_cast<char *>(feature.data()), feature_size)) {
      std::cerr << "Failed to read from " << query_file << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    query_points.feature_vector(id).insert(
        query_points.feature_vector(id).begin(), feature.begin(),
        feature.end());
  }
}

template <typename id_t, typename distance_t>
inline void read_ground_truths(std::string_view                      file_name,
                               std::vector<std::vector<id_t>>       &friends,
                               std::vector<std::vector<distance_t>> &distances,
                               const bool                            verbose) {
  std::ifstream ifs(file_name.data(), std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << file_name << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  uint32_t num_queries = 0;
  uint32_t num_friends = 0;
  if (!ifs.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries)) ||
      !ifs.read(reinterpret_cast<char *>(&num_friends), sizeof(num_friends))) {
    std::cerr << "Failed to read from " << file_name << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (verbose) {
    std::cout << "#of point\t" << num_queries << std::endl;
    std::cout << "#of friends\t" << num_friends << std::endl;
  }

  const auto friends_size = num_friends * sizeof(id_t);
  friends.emplace_back(friends_size);

  const auto distances_size = num_friends * sizeof(distance_t);
  distances.emplace_back(distances_size);

  for (id_t id = 0; id < num_queries; ++id) {
    if (!ifs.read(reinterpret_cast<char *>(friends.back().data()),
                  friends_size) ||
        !ifs.read(reinterpret_cast<char *>(distances.back().data()),
                  distances_size)) {
      std::cerr << "Failed to read from " << file_name << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
}
}  // namespace saltatlas::ann_bench_data_reader

// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include <string>
#include <fstream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
#include <cassert>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cstring>

#include "saltatlas/neo_dnnd/mpi.hpp"
#include "saltatlas/neo_dnnd/omp.hpp"
#include "saltatlas/neo_dnnd/string_cast.hpp"

namespace saltatlas::neo_dnnd {

template <typename _id_type, typename _feature_elem_type>
class dataset_reader {
 public:
  using id_type = _id_type;
  using feature_elem_type = _feature_elem_type;

  dataset_reader() = default;
  ~dataset_reader() = default;

  template <typename partitioner_type>
  static std::vector<std::pair<id_type, std::vector<feature_elem_type>>> read(
      const std::filesystem::path& path, const std::string_view format,
      const partitioner_type& partitioner, mpi::communicator& comm) {
    if (format == "wsv") {
      return priv_read_wsv(priv_get_file_list(path), partitioner, comm);
    } else if (format == "wsv-id") {
      return priv_read_wsv_with_id(priv_get_file_list(path), partitioner, comm);
    } else {
      comm.cerr0() << "Unknown dataset format: " << format << std::endl;
      comm.abort();
    }
    return {};
  }

 private:
  static std::vector<std::filesystem::path> priv_get_file_list(
      const std::filesystem::path& path) {
    std::vector<std::filesystem::path> file_list;
    // If 'path' is a file, return it.
    // If 'path' is a directory, return all files in it.
    if (std::filesystem::is_regular_file(path)) {
      file_list.push_back(path);
    } else if (std::filesystem::is_directory(path)) {
      for (const auto& entry :
           std::filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
          file_list.push_back(entry.path());
        }
      }
    }
    return file_list;
  }

  template <typename partitioner_type>
  static auto priv_read_wsv(const std::vector<std::filesystem::path>& paths,
                            const partitioner_type& partitioner,
                            mpi::communicator& comm) {
    const std::vector<std::size_t> offsets =
        priv_convert_to_offsets(priv_count_lines_in_files(paths, comm));

    comm.cout0() << "Reading points w/o ID..." << std::endl;
    std::vector<std::vector<id_type>> read_ids(comm.size());
    std::vector<std::vector<feature_elem_type>> read_features(comm.size());
    std::size_t dims = 0;
    for (int i = 0; i < int(paths.size()); ++i) {
      if (i % int(comm.size()) != comm.rank()) {
        continue;
      }
      std::ifstream ifs(paths[i]);
      if (ifs) {
        // std::cout << "Opened " << paths[i] << std::endl;
      } else {
        std::cerr << "Failed to open " << paths[i] << std::endl;
        comm.abort();
      }

      id_type pid = offsets[i];
      std::string line;
      while (std::getline(ifs, line)) {
        auto feature = ndndtl::str_split<feature_elem_type>(line);
        if (dims == 0) {
          dims = feature.size();
        } else {
          if (dims != feature.size()) {
            std::cerr << "Inconsistent number of dimensions: " << dims << " vs "
                      << feature.size() << std::endl;
            comm.abort();
          }
        }
        read_ids[partitioner(pid)].push_back(pid);
        read_features[partitioner(pid)].insert(
            read_features[partitioner(pid)].end(), feature.begin(),
            feature.end());
        ++pid;
      }
    }
    comm.barrier();

    comm.bcast(dims, 0);

    return priv_distribute_points(read_ids, read_features, dims, comm);
  }

  template <typename partitioner_type>
  static auto priv_read_wsv_with_id(
      const std::vector<std::filesystem::path>& paths,
      const partitioner_type& partitioner, mpi::communicator& comm) {

    std::vector<std::vector<id_type>> read_ids(comm.size());
    std::vector<std::vector<feature_elem_type>> read_features(comm.size());
    std::size_t dims = 0;
    std::unordered_set<id_type> id_set;

    comm.cout0() << "Reading points with ID..." << std::endl;
    for (int i = 0; i < int(paths.size()); ++i) {
      if (i % comm.size() != comm.rank()) {
        continue;
      }
      std::ifstream ifs(paths[i]);
      if (ifs) {
        // std::cout << "Opened " << paths[i] << std::endl;
      } else {
        std::cerr << "Failed to open " << paths[i] << std::endl;
        comm.abort();
      }

      std::string line;
      while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        id_type pid;
        ss >> pid;
        if (id_set.find(pid) != id_set.end()) {
          std::cerr << "Duplicate ID: " << pid << std::endl;
          comm.abort();
        }
        line = ss.str().substr(ss.tellg());  // Remove the ID part.
        auto feature = ndndtl::str_split<feature_elem_type>(line);
        if (dims == 0) {
          dims = feature.size();
        } else {
          if (dims != feature.size()) {
            std::cerr << "Inconsistent number of dimensions: " << dims << " vs "
                      << feature.size() << std::endl;
            comm.abort();
          }
        }
        const auto owner = partitioner(pid);
        read_ids[owner].push_back(pid);
        read_features[owner].insert(read_features[owner].end(), feature.begin(),
                                    feature.end());
      }
    }
    comm.barrier();

    comm.bcast(dims, 0);

    return priv_distribute_points(read_ids, read_features, dims, comm);
  }

  static std::vector<std::size_t> priv_count_lines_in_files(
      const std::vector<std::filesystem::path>& paths,
      mpi::communicator& comm) {
    comm.cout0() << "Counting data..." << std::endl;

    std::vector<std::size_t> num_lines(paths.size());
    for (int i = 0; i < int(paths.size()); ++i) {
      if (i % comm.size() != comm.rank()) {
        continue;
      }
      std::ifstream ifs(paths[i]);
      if (ifs) {
        // std::cout << "Opened " << paths[i] << std::endl;
      } else {
        std::cerr << "Failed to open " << paths[i] << std::endl;
        comm.abort();
      }

      std::string line;
      while (std::getline(ifs, line)) {
        ++num_lines[i];
      }
    }
    comm.barrier();

    // broadcast the number of lines in each file.
    for (std::size_t i = 0; i < paths.size(); ++i) {
      comm.bcast(num_lines[i], i % comm.size());
    }

    comm.cout0() << "#of total lines: " << num_lines.back() << std::endl;
    if (std::numeric_limits<id_type>::max() < num_lines.back()) {
      comm.cerr0() << "Too small ID type: " << typeid(id_type).name()
                   << std::endl;
      comm.abort();
    }

    return num_lines;
  }

  static std::vector<std::size_t> priv_convert_to_offsets(
      const std::vector<std::size_t>& num_lines) {
    std::vector<std::size_t> offsets(num_lines.size());
    std::partial_sum(num_lines.cbegin(), num_lines.cend(), offsets.begin());
    offsets.insert(offsets.begin(), 0);
    return offsets;
  }

  static auto priv_distribute_points(
      const std::vector<std::vector<id_type>>& read_ids,
      const std::vector<std::vector<feature_elem_type>>& read_features,
      const std::size_t dims, mpi::communicator& comm) {
    assert(read_ids.size() == std::size_t(comm.size()));
    assert(read_features.size() == std::size_t(comm.size()));

    comm.cout0() << "Distributing read points..." << std::endl;

    std::size_t num_assigned_points = 0;
    for (int r = 0; r < comm.size(); ++r) {
      std::size_t n = read_ids[r].size();
      DNND2_CHECK_MPI(::MPI_Reduce(
          &n, (r == comm.rank()) ? &num_assigned_points : nullptr, 1,
          mpi::data_type::get<std::size_t>(), MPI_SUM, r, comm.comm()));
    }

    std::vector<std::pair<id_type, std::vector<feature_elem_type>>>
        assigned_points;
    assigned_points.reserve(num_assigned_points);

    mpi::pair_wise_all_to_all(
        comm.size(), comm.rank(),
        [&](const int pair_rank) {
          std::vector<id_type> ids_recv_buf;
          comm.sendrecv_arb_size(pair_rank, read_ids[pair_rank], ids_recv_buf);

          std::vector<feature_elem_type> features_recv_buf;
          comm.sendrecv_arb_size(pair_rank, read_features[pair_rank],
                                 features_recv_buf);

          const auto num_recv_points = ids_recv_buf.size();
          for (std::size_t i = 0; i < num_recv_points; ++i) {
            assigned_points.emplace_back(std::make_pair(
                ids_recv_buf[i], std::vector<feature_elem_type>(dims)));
            auto& f = assigned_points.back().second;
            assert(f.size() == dims);
            std::memcpy(f.data(), features_recv_buf.data() + i * dims,
                        dims * sizeof(feature_elem_type));
          }
        },
        comm.comm());
    comm.barrier();
    return assigned_points;
  }
};
}  // namespace saltatlas::neo_dnnd
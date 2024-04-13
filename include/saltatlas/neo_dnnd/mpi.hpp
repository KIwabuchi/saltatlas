// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <optional>

#include <mpi.h>

#include "saltatlas/neo_dnnd/utility.hpp"

#define DNND2_CHECK_MPI(ret)                                                  \
  do {                                                                        \
    if (ret != MPI_SUCCESS) {                                                 \
      std::cerr << __FILE__ << ":" << __LINE__ << " MPI error." << std::endl; \
      ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                              \
    }                                                                         \
  } while (0)

namespace saltatlas::neo_dnnd::mpi {

/// \brief Returns the MPI_Datatype for the given type.
struct data_type {
  template <typename T>
  static constexpr MPI_Datatype get() {
    if constexpr (std::is_same_v<T, char>) {
      return MPI_CHAR;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
      return MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, int>) {
      return MPI_INT;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
      return MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, long>) {
      return MPI_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long>) {
      return MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T, long long> ||
                         std::is_same_v<T, long long int> ||
                         std::is_same_v<T, long long unsigned int> ||
                         std::is_same_v<T, std::size_t>) {
      return MPI_LONG_LONG_INT;
    } else if constexpr (std::is_same_v<T, float>) {
      return MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
      return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
      return MPI_LONG_DOUBLE;
    }
    return MPI_DATATYPE_NULL;
  }
};

class communicator {
 public:
  explicit communicator(MPI_Comm comm = MPI_COMM_WORLD) : m_comm(comm) {
    DNND2_CHECK_MPI(::MPI_Comm_rank(m_comm, &m_rank));
    DNND2_CHECK_MPI(::MPI_Comm_size(m_comm, &m_size));
    priv_get_local_comm_info();
  }

  communicator(const communicator&) = delete;
  communicator& operator=(const communicator&) = delete;
  communicator(communicator&&) = delete;
  communicator& operator=(communicator&&) = delete;

  ~communicator() {
    DNND2_CHECK_MPI(::MPI_Comm_free(&m_local_comm));
  }

  int rank() const { return m_rank; }
  int size() const { return m_size; }

  int local_rank() const { return m_local_rank; }
  int local_size() const { return m_local_size; }

  int num_nodes() const {
    assert(m_size % m_local_size == 0);
    return m_size / m_local_size;
  }

  int node_rank() const { return m_rank / m_local_size; }

  /// \brief Returns std::cout if the rank is 0.
  /// \return std::cout if the rank is 0, otherwise std::nullopt.
  std::ostream& cout0() const {
    static std::ostringstream dummy;
    if (m_rank == 0) {
      return std::cout;
    } else {
      return dummy;
    }
  }

  std::ostream& cerr0() const {
    static std::ostringstream dummy;
    if (m_rank == 0) {
      return std::cerr;
    } else {
      return dummy;
    }
  }

  std::ostream& cout() const {
    std::cout << "[" << m_rank << "]: ";
    return std::cout;
  }

  std::ostream& cerr() const {
    std::cout << "[" << m_rank << "]: ";
    return std::cout;
  }

  void barrier() const { DNND2_CHECK_MPI(::MPI_Barrier(m_comm)); }

  void local_barrier() const { DNND2_CHECK_MPI(::MPI_Barrier(m_local_comm)); }

  void abort() const { DNND2_CHECK_MPI(::MPI_Abort(m_comm, EXIT_FAILURE)); }

  MPI_Comm comm() const { return m_comm; }

  template <typename T>
  void isend(const std::vector<T>& buf, const int dest,
             MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Isend(buf.data(), buf.size(), data_type::get<T>(),
                                dest, 0, m_comm, &request));
  }

  template <typename T>
  void isend(const T* const buf, const int count, ::MPI_Datatype& type,
             const int dest, MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Isend(buf, count, type, dest, 0, m_comm, &request));
  }

  template <typename T>
  void irecv(const int from, std::vector<T>& buf, MPI_Request& request) const {
    const auto count = priv_get_count(from, data_type::get<T>());
    buf.resize(count);
    DNND2_CHECK_MPI(::MPI_Irecv(buf.data(), count, data_type::get<T>(), from, 0,
                                m_comm, &request));
  }
  template <typename T>
  void irecv(const int from, std::vector<T>& buf, ::MPI_Datatype& type,
             MPI_Request& request) const {
    const auto count = priv_get_count(from, type);
    buf.resize(count);
    DNND2_CHECK_MPI(
        ::MPI_Irecv(buf.data(), count, type, from, 0, m_comm, &request));
  }

  void wait(MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Wait(&request, MPI_STATUS_IGNORE));
  }

  template <typename T>
  void all_reduce(const T& send, T& recv, MPI_Op op) const {
    DNND2_CHECK_MPI(
        ::MPI_Allreduce(&send, &recv, 1, data_type::get<T>(), op, m_comm));
  }

  template <typename T>
  T all_reduce_sum(const T send) const {
    T sum;
    all_reduce(send, sum, MPI_SUM);
    return sum;
  }

  template <typename T>
  T all_reduce_min(const T send) const {
    T min;
    all_reduce(send, min, MPI_MIN);
    return min;
  }

  template <typename T>
  T all_reduce_max(const T send) const {
    T max;
    all_reduce(send, max, MPI_MAX);
    return max;
  }

  template <typename T>
  T all_node_reduce_sum(const T send) const {
    T sum;
    const T buf = local_rank() == 0 ? send : 0;
    all_reduce(buf, sum, MPI_SUM);
    return sum;
  }

  template <typename T>
  T all_node_reduce_max(const T send) const {
    T max;
    const T buf = local_rank() == 0 ? send : std::numeric_limits<T>::min();
    all_reduce(buf, max, MPI_MAX);
    return max;
  }

  template <typename T>
  T all_node_reduce_min(const T send) const {
    T min;
    const T buf = local_rank() == 0 ? send : std::numeric_limits<T>::max();
    all_reduce(buf, min, MPI_MIN);
    return min;
  }

  template <typename T>
  std::vector<T> reduce_sum(const std::vector<T>& send, const int root) const {
    std::vector<T> recv;
    recv.resize(send.size());
    DNND2_CHECK_MPI(::MPI_Reduce(send.data(), recv.data(), send.size(),
                                 data_type::get<T>(), MPI_SUM, root, m_comm));
    return recv;
  }

  template <class T>
  void all_gather(const T& send, std::vector<T>& recv_buf) const {
    recv_buf.resize(m_size);
    all_gather(send, recv_buf.data());
  }

  template <class T>
  void all_gather(const T& send, T* const recv_buf) const {
    DNND2_CHECK_MPI(::MPI_Allgather(&send, 1, data_type::get<T>(), recv_buf, 1,
                                    data_type::get<T>(), m_comm));
  }

  // Not tested yet
  template <class T>
  void all_gather_v(const T* const send_buf, const std::size_t send_count,
                    std::vector<T>& recv_buf) const {
    std::vector<T> recv_counts(m_size);
    all_gather(send_count, recv_counts.data());

    std::vector<int> displs(m_size);
    displs[0] = 0;
    for (int i = 1; i < m_size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
    recv_buf.resize(displs.back());
    DNND2_CHECK_MPI(::MPI_Allgatherv(
        send_buf, send_count, data_type::get<T>(), recv_buf.data(),
        recv_counts.data(), displs.data(), data_type::get<T>(), m_comm));
  }

  template <class T>
  void bcast(T& data, const int root) const {
    DNND2_CHECK_MPI(::MPI_Bcast(&data, 1, data_type::get<T>(), root, m_comm));
  }

  /// \brief Send and receive data of arbitrary size to and from a pair rank.
  /// If the size of the data is larger than 'batch_size_byte',
  /// the data is sent in batches.
  template <typename T>
  void sendrecv_arb_size(const int pair_rank, const std::vector<T>& send_buffer,
                         std::vector<T>& recv_buffer,
                         ::MPI_Datatype data_type = data_type::get<T>(),
                         const std::size_t batch_size_byte = 1 << 26) {
    if (pair_rank == m_rank) {
      recv_buffer = send_buffer;  // TODO: avoid copy?
      return;
    }

    const std::size_t batch_size = batch_size_byte / sizeof(T);
    const int num_batches = (send_buffer.size() + batch_size - 1) / batch_size;
    bool received_all = false;
    recv_buffer.clear();
    for (int i = 0; i < num_batches || !received_all; ++i) {
      const bool reached_last = i >= num_batches - 1;
      const auto off = std::min(batch_size * i, send_buffer.size());
      const auto send_size = std::min(batch_size, send_buffer.size() - off);
      received_all |= priv_sendrecv_arb_size_helper(
          pair_rank, send_buffer.data() + off, send_size, reached_last,
          data_type, recv_buffer);
    }
  }

  /// \brief sendrecv_arb_size optimized version.
  template <typename T>
  void sendrecv_arb_size_opt(const int pair_rank, std::vector<T>& send_buffer,
                             std::vector<T>& recv_buffer,
                             ::MPI_Datatype data_type = data_type::get<T>(),
                             const std::size_t batch_size_byte = 1 << 26) {
    if (pair_rank == m_rank) {
      recv_buffer = std::move(send_buffer);
      return;
    }
    sendrecv_arb_size(pair_rank, send_buffer, recv_buffer, data_type,
                      batch_size_byte);
  }

  template <typename T>
  void all_to_all(const std::vector<T>& send_buffer,
                  std::vector<T>& recv_buffer) {
    if (send_buffer.size() * sizeof(T) > std::numeric_limits<int>::max()) {
      std::cerr << "Too large data size to send: "
                << send_buffer.size() * sizeof(T) << std::endl;
      abort();
    }

    recv_buffer.resize(send_buffer.size());
    DNND2_CHECK_MPI(::MPI_Alltoall(
        send_buffer.data(), send_buffer.size(), data_type::get<T>(),
        recv_buffer.data(), send_buffer.size(), data_type::get<T>(), m_comm));
  }

  template <typename T>
  void all_to_all_v(const std::vector<T>& send_buffer,
                    const std::vector<int>& send_counts,
                    std::vector<T>& recv_buffer) {
    if (send_buffer.size() * sizeof(T) > (1ULL << 31ULL)) {
      std::cerr << "Too large data size to send: "
                << send_buffer.size() * sizeof(T) << std::endl;
      abort();
    }
    assert(send_counts.size() == (std::size_t)m_size);

    auto sdispls = send_counts;
    std::partial_sum(sdispls.cbegin(), sdispls.cend(), sdispls.begin());
    sdispls.insert(sdispls.begin(), 0);

    std::vector<int> recv_counts;
    all_to_all(send_counts, recv_counts);

    auto rdispls = recv_counts;
    std::partial_sum(rdispls.cbegin(), rdispls.cend(), rdispls.begin());
    rdispls.insert(rdispls.begin(), 0);
    recv_buffer.resize(rdispls.back());

    DNND2_CHECK_MPI(::MPI_Alltoallv(
        send_buffer.data(), send_counts.data(), sdispls.data(),
        data_type::get<T>(), recv_buffer.data(), recv_counts.data(),
        rdispls.data(), data_type::get<T>(), m_comm));
  }

  template <typename T>
  void all_to_all_v(const std::vector<std::vector<T>>& to_send,
                    std::vector<T>& recv_buffer) {
    std::vector<int> send_counts(m_size, 0);
    std::size_t num_send = 0;
    for (int r = 0; r < to_send.size(); ++r) {
      send_counts[r] = to_send[r].size();
      num_send += to_send[r].size();
    }

    std::vector<T> send_buf(num_send);
    for (auto& row : to_send) {
      send_buf.insert(send_buf.end(), row.begin(), row.end());
    }
    all_gather_v(send_buf, send_counts, recv_buffer);
  }

 private:
  void priv_get_local_comm_info() {
    DNND2_CHECK_MPI(::MPI_Comm_split_type(m_comm, MPI_COMM_TYPE_SHARED, m_rank,
                                          MPI_INFO_NULL, &m_local_comm));
    DNND2_CHECK_MPI(::MPI_Comm_size(m_local_comm, &m_local_size));
    DNND2_CHECK_MPI(::MPI_Comm_rank(m_local_comm, &m_local_rank));
  }

  int priv_get_count(const int from, ::MPI_Datatype type) const {
    MPI_Status status;
    DNND2_CHECK_MPI(::MPI_Probe(from, MPI_ANY_TAG, m_comm, &status));
    int count;
    DNND2_CHECK_MPI(::MPI_Get_count(&status, type, &count));
    return count;
  }

  template <typename T>
  int priv_sendrecv_arb_size_helper(const int pair_rank,
                                    const T* const send_buffer,
                                    const int send_count,
                                    const bool reached_last,
                                    ::MPI_Datatype data_type,
                                    std::vector<T>& recv_buffer) {
    MPI_Request isend_request;
    if (send_count * sizeof(T) > (1ULL << 31ULL)) {
      std::cerr << "Too large data size to send: " << send_count * sizeof(T)
                << std::endl;
      abort();
    }
    DNND2_CHECK_MPI(::MPI_Isend(send_buffer, send_count, data_type, pair_rank,
                                reached_last ? 1 : 0, m_comm, &isend_request));

    MPI_Status status;
    DNND2_CHECK_MPI(::MPI_Probe(pair_rank, MPI_ANY_TAG, m_comm, &status));
    int count;
    DNND2_CHECK_MPI(::MPI_Get_count(&status, data_type, &count));
    assert(count >= 0);

    const auto off = recv_buffer.size();
    recv_buffer.resize(count + off);

    DNND2_CHECK_MPI(::MPI_Recv(recv_buffer.data() + off, count, data_type,
                               pair_rank, MPI_ANY_TAG, m_comm,
                               MPI_STATUS_IGNORE));
    DNND2_CHECK_MPI(::MPI_Wait(&isend_request, MPI_STATUS_IGNORE));

    const bool received_last = (status.MPI_TAG == 1);

    return received_last;
  }

  ::MPI_Comm m_comm;
  ::MPI_Comm m_local_comm;
  int m_rank;
  int m_size;
  int m_local_rank;
  int m_local_size;
};

/// \brief Execute a user-defined function for each unique pair of ranks.
// The user-defined function is executed as 'size' times (including
// self-directed communication). In every execution, each rank is exclusively
// paired with one other rank. If rank ‘a' is paired with rank ‘b', rank ‘b' is
// paired with only rank ‘a' during the same step. Thus, all ranks can execute
// the function, utilizing the parallelism fully.
/// \tparam function_t Function type.
/// \param comm_size MPI size. Must be a power of 2.
/// \param comm_rank My MPI rank.
/// \param func Function to execute.
template <typename function_t>
inline constexpr void pair_wise_all_to_all(
    const int comm_size, const int comm_rank, const function_t& func,
    const MPI_Comm comm = MPI_COMM_WORLD) {
  // make sure comm_size is a power of 2
  if ((comm_size & (comm_size - 1)) != 0) {
    if (comm_rank == 0) {
      std::cerr << "MPI size must be a power of 2" << std::endl;
      DNND2_CHECK_MPI(::MPI_Abort(comm, EXIT_FAILURE));
    }
  }

  func(comm_rank);  // self-directed communication

  for (int block_size = 1; block_size <= comm_size / 2; block_size *= 2) {
    const bool left_block = (comm_rank / block_size) % 2 == 0;
    for (int i = 0; i < block_size; ++i) {
      const int pair_blocK_begin =
          (left_block)
              ? saltatlas::neo_dnnd::utility::round_down(comm_rank + block_size, block_size)
              : saltatlas::neo_dnnd::utility::round_down(comm_rank - block_size, block_size);
      const int in_block_pos = (left_block) ? (comm_rank + i) % block_size
                                            : (comm_rank - i) % block_size;
      const int pair_rank = pair_blocK_begin + in_block_pos;
      func(pair_rank);
    }
  }
}

inline std::vector<int> get_pair_wise_all_to_all_pattern(const int comm_size,
                                                         const int comm_rank) {
  std::vector<int> table;
  pair_wise_all_to_all(comm_size, comm_rank, [&](const int pair_rank) {
    table.push_back(pair_rank);
  });
  return table;
}

inline void show_task_distribution(const std::vector<std::size_t>& table) {
  const auto sum = std::accumulate(table.begin(), table.end(), std::size_t(0));
  const auto mean = (double)sum / (double)table.size();
  std::cout << "Assigned " << sum << " tasks to " << table.size() << " workers"
            << std::endl;
  std::cout << "Max, Mean, Min:\t"
            << "" << *std::max_element(table.begin(), table.end()) << ", "
            << mean << ", " << *std::min_element(table.begin(), table.end())
            << std::endl;
  double x = 0;
  for (const auto n : table) x += std::pow(n - mean, 2);
  const auto dv = std::sqrt(x / table.size());
  std::cout << "Standard Deviation " << dv << std::endl;
}

/// \brief Compute the number of tasks each MPI rank works on.
/// \param num_local_tasks #of tasks in local.
/// \param batch_size Global batch size. Up to this number of tasks are
/// assigned over all ranks. If 0 is specified, all tasks are assigned. \param
/// mpi_rank My MPI rank. \param mpi_size MPI size. \param verbose Verbose
/// mode. \return #of tasks assigned to myself.
inline std::size_t assign_tasks(const std::size_t num_local_tasks,
                                const std::size_t batch_size,
                                const int mpi_rank, const int mpi_size,
                                const bool verbose,
                                const MPI_Comm mpi_comm = MPI_COMM_WORLD) {
  if (batch_size == 0) {
    return num_local_tasks;
  }

  std::size_t local_num_assigned_tasks = 0;
  if (mpi_rank > 0) {
    // Send the number of tasks to process to rank 0.
    DNND2_CHECK_MPI(
        ::MPI_Send(&num_local_tasks, 1, MPI_UNSIGNED_LONG, 0, 0, mpi_comm));

    // Receive the number of assigned tasks to process from rank 0.
    MPI_Status status;
    DNND2_CHECK_MPI(::MPI_Recv(&local_num_assigned_tasks, 1, MPI_UNSIGNED_LONG,
                               0, 0, mpi_comm, &status));
  } else {
    // Gather the number of tasks each MPI has
    std::vector<std::size_t> num_remaining_tasks_table(mpi_size, 0);
    num_remaining_tasks_table[0] = num_local_tasks;
    for (int r = 1; r < mpi_size; ++r) {
      MPI_Status status;
      DNND2_CHECK_MPI(::MPI_Recv(&num_remaining_tasks_table[r], 1,
                                 MPI_UNSIGNED_LONG, r, 0, mpi_comm, &status));
    }

    const auto num_global_tasks =
        std::accumulate(num_remaining_tasks_table.begin(),
                        num_remaining_tasks_table.end(), std::size_t(0));
    assert(batch_size > 0);
    std::size_t num_global_unassigned_tasks =
        std::min(batch_size, num_global_tasks);

    // Assigned tasks
    std::vector<std::size_t> task_assignment_table(mpi_size, 0);
    while (num_global_unassigned_tasks > 0) {
      const std::size_t max_num_tasks_per_rank =
          (num_global_unassigned_tasks < (std::size_t)mpi_size)
              ? 1
              : (num_global_unassigned_tasks + mpi_size - 1) / mpi_size;
      for (std::size_t r = 0; r < num_remaining_tasks_table.size(); ++r) {
        const auto n =
            std::min({max_num_tasks_per_rank, num_remaining_tasks_table[r],
                      num_global_unassigned_tasks});
        num_remaining_tasks_table[r] -= n;
        task_assignment_table[r] += n;
        num_global_unassigned_tasks -= n;
      }
    }

    // Tell the computed numbers to the other ranks
    for (int r = 1; r < mpi_size; ++r) {
      DNND2_CHECK_MPI(::MPI_Send(&task_assignment_table[r], 1,
                                 MPI_UNSIGNED_LONG, r, 0, mpi_comm));
    }
    local_num_assigned_tasks = task_assignment_table[0];

    if (verbose) {
      const auto n =
          std::accumulate(task_assignment_table.begin(),
                          task_assignment_table.end(), (std::size_t)0);
      std::cout << "#of total task\t" << num_global_tasks << std::endl;
      std::cout << "#of total assigned task\t" << n << std::endl;
      std::cout << "#of unassigned tasks\t" << num_global_tasks - n
                << std::endl;
      show_task_distribution(task_assignment_table);
    }
  }

  assert(local_num_assigned_tasks <= num_local_tasks);

  DNND2_CHECK_MPI(::MPI_Barrier(mpi_comm));

  return local_num_assigned_tasks;
}

}  // namespace saltatlas::neo_dnnd::mpi
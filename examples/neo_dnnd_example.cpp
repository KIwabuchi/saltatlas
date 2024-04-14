// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Must run srun with '-mblock' option, which is the default one.
// Do not use '--mpibind=off' option.

#define METALL_DISABLE_CONCURRENCY

#ifdef __APPLE__
#define METALL_DEFAULT_CAPACITY (1ULL << 30ULL)
#else
#define METALL_DEFAULT_CAPACITY (1ULL << 36ULL)
#endif

#include <filesystem>
#include <iostream>
#include <random>
#include <string>

#include "saltatlas/neo_dnnd/dataset_reader.hpp"
#include "saltatlas/neo_dnnd/neo_dnnd.hpp"

#ifndef NDEBUG
#include "saltatlas/neo_dnnd/backtrace.hpp"
#endif

using namespace saltatlas::neo_dnnd;

using id_type = uint32_t;
#ifdef SALTATLAS_NEO_DNND_FEATURE_ELEMENT_TYPE
using feature_elem_type = SALTATLAS_NEO_DNND_FEATURE_ELEMENT_TYPE;
#else
using feature_elem_type = float;
#endif
using distance_type =
    std::conditional_t<std::is_same_v<feature_elem_type, double>, double,
                       float>;

using kernel = dnnd_kernel<id_type, feature_elem_type, distance_type>;

template <typename knng_type>
void dump_knng(const std::filesystem::path& base_path, const knng_type& knng,
               const int rank, bool dump_distance = false) {
  std::filesystem::path knng_out_path =
      base_path.string() + "-" + std::to_string(rank) + ".txt";

  std::ofstream ofs(knng_out_path);

  if (!ofs.is_open()) {
    std::cerr << "Failed to create kNNG file" << std::endl;
    return;
  }

  for (const auto& elem : knng) {
    ofs << elem.first;
    for (const auto& neighbor : elem.second) {
      ofs << " " << neighbor.id;
    }
    ofs << "\n";

    if (!dump_distance) continue;
    ofs << "0.0";  // dummy
    for (const auto& neighbor : elem.second) {
      ofs << " " << neighbor.distance;
    }
    ofs << "\n";
  }
  ofs.close();
}

struct options {
  std::string dataset_path;
  std::string dataset_format;
  std::string distance_function;
  int         k{0};
  double      rho   = 0.8;
  double      delta = 0.001;
  std::string knng_dump_dir;
  bool        optimize       = false;
  double      pruning_factor = -1;
  std::size_t batch_size     = 1ULL << 25;
  // #of fvs to cache per rank, 0 means no caching.
  double popular_fv_ratio              = 0.0f;
  bool   donot_remove_dup_fvs          = false;
  bool   donot_share_pstore_regionally = false;
  bool   dump_distance                 = false;
  bool   verbose                       = false;

  template <typename out_stream_type>
  void show(out_stream_type& os) {
    os << "Options" << std::endl;
    os << "  Dataset: " << dataset_path << std::endl;
    os << "  Dataset format: " << dataset_format << std::endl;
    os << "  Distance function: " << distance_function << std::endl;
    os << "  k: " << k << std::endl;
    os << "  rho: " << rho << std::endl;
    os << "  delta: " << delta << std::endl;
    os << "  Out dir: " << knng_dump_dir << std::endl;
    os << "  Dump distance: " << dump_distance << std::endl;
    os << "  Optimize: " << optimize << std::endl;
    os << "  Optimization pruning factor: " << pruning_factor << std::endl;
    os << "  Batch size: " << batch_size << std::endl;
    os << "  Ratio of FVs to replicate: " << popular_fv_ratio << std::endl;
    os << "  Do not remove duplicate feature vectors: " << donot_remove_dup_fvs
       << std::endl;
    os << "  Do not share point store regionally: "
       << donot_share_pstore_regionally << std::endl;
    os << "  Verbose: " << verbose << std::endl;
  }
};

template <typename ost>
void usage(ost& os) {
  // Show detailed usage
  os << "Usage: neo_dnnd [options]" << std::endl;
  os << "Options:"
     << "\n -i [string, required] Path to the dataset file or directory."
     << "\n -p [string, required] Dataset file format:"
     << "\n \t'wsv' (whitespace-separated values w/o ID),"
     << "\n \t'wsv-id' (WSV format and the first column is point ID)."
     << "\n -f [string, required] Distance function:"
     << "\n \t'l2' (L2 distance), "
        "\n \t'sql2' (squared L2, faster one), "
        "\n \t'cosine' (cosine similarity), "
        "\n \t'altcosine' (alternative faster cosine similarity), "
        "\n \t'jaccard' (Jaccard index), "
        "\n \t'altjaccard' (alternative faster Jaccard index), "
        "\n \tor 'levenshtein' (Levenshtein distance)."
     << "\n -k [int, required] k for KNNG construction."
     << "\n -r [double, optional] rho (sampling) parameter in NN-Descent. "
        "Default: 0.8."
     << "\n -d [double, optional] delta (terminal condition) parameter in "
        "NN-Descent. Default: 0.001."
     << "\n -b [int, optional] KNNG construction batch size. Default: 2^25."
     << "\n -L [optional] Do not share point store in local node."
     << "\n -A [optional] Do not remove duplicate feature vectors."
     << "\n -P [double, optional] Ratio of FVs to replicate. Between 0 and "
        "1.0. Default: 0."
     << "\n -O [optional] Optimize KNNG after construction."
     << "\n -m [double, optional] High-degree edge pruning factor for "
        "optimization.  Default: -1 (no pruning)."
     << "\n -G [string, optional] Directory to dump KNNG. Default: no dump."
     << "\n -D [optional] Dump distance to output KNNG files."
     << "\n -v [flag, optional] Verbose mode." << std::endl;
}

bool parse_options(int argc, char* argv[], options& opt, bool& show_usage) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:r:d:b:G:AP:LSOm:Dvh")) != -1) {
    switch (p) {
      case 'i':
        opt.dataset_path = optarg;
        break;
      case 'p':
        opt.dataset_format = optarg;
        break;
      case 'f':
        opt.distance_function = optarg;
        break;
      case 'k':
        opt.k = std::stoi(optarg);
        break;
      case 'r':
        opt.rho = std::stod(optarg);
        break;
      case 'd':
        opt.delta = std::stod(optarg);
        break;
      case 'b':
        opt.batch_size = std::stoi(optarg);
        break;
      case 'G':
        opt.knng_dump_dir = optarg;
        break;
      case 'A':
        opt.donot_remove_dup_fvs = true;
        break;
      case 'P':
        opt.popular_fv_ratio = std::stod(optarg);
        break;
      case 'L':
        opt.donot_share_pstore_regionally = true;
        break;
      case 'S':;  // Do nothing
        break;
      case 'O':
        opt.optimize = true;
        break;
      case 'm':
        opt.pruning_factor = std::stod(optarg);
        break;
      case 'D':
        opt.dump_distance = true;
        break;
      case 'v':
        opt.verbose = true;
        break;
      case 'h':
        show_usage = true;
        return true;
      default:
        return false;
    }
  }

  return true;
}

void show_mpi_info(mpi::communicator& comm) {
  comm.cout0() << "MPI Info" << std::endl;
  comm.cout0() << "  #of comm ranks: " << comm.size() << std::endl;
  comm.cout0() << "  #of comp. nodes: " << comm.num_nodes() << std::endl;
  comm.cout0() << "  Node size: " << comm.local_size() << std::endl;
  comm.cout0() << "  #of numa nodes: " << numa::get_num_avail_nodes()
               << std::endl;
  comm.cout0() << "  Rank\tLocal rank" << std::endl;
  for (int i = 0; i < comm.size(); ++i) {
    comm.barrier();
    if (i == comm.rank()) {
      std::cout << "  \t" << comm.rank() << "\t" << comm.local_rank()
                << std::endl;
    }
  }
  comm.barrier();
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
  signal(SIGSEGV, show_backtrace);
#endif

  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    mpi::communicator comm;
    const int         mpi_rank = comm.rank();
    const int         mpi_size = comm.size();

    comm.cout0() << "========================================" << std::endl;
    comm.cout0() << "Start NEO-DNND" << std::endl;
    comm.cout0() << "========================================" << std::endl;

    if (provided < MPI_THREAD_FUNNELED) {
      comm.cerr0()
          << "The threading support level is lesser than that demanded."
          << std::endl;
      comm.abort();
    }

    options opt;
    bool    show_usage = false;
    if (!parse_options(argc, argv, opt, show_usage)) {
      usage(comm.cerr0());
      return EXIT_FAILURE;
    }
    if (show_usage) {
      usage(comm.cout0());
      goto EXIT_NORMAL;
    }
    opt.show(comm.cout0());
    if (opt.verbose) {
      {
        omp::set_num_threads(2);
        OMP_DIRECTIVE(parallel) {
          const auto num_threads = omp::get_num_threads();
          OMP_DIRECTIVE(single) {
            comm.cout0() << "#of threads: " << num_threads << std::endl;
          }
        }
      }
      comm.barrier();
      show_mpi_info(comm);
    }
    comm.barrier();

    time_recorder recorder;
    {
      kernel kernel(
          distance::similarity_function<feature_elem_type, distance_type>(
              opt.distance_function),
          recorder, comm, opt.verbose);
      comm.barrier();

      comm.cout0() << "\n========================================" << std::endl;
      comm.cout0() << "Read dataset" << std::endl;
      comm.cout0() << "========================================" << std::endl;
      recorder.start("read_dataset");
      kernel.read_dataset(opt.dataset_path, opt.dataset_format,
                          !opt.donot_share_pstore_regionally);
      recorder.stop();

      comm.cout0() << "\n========================================" << std::endl;
      comm.cout0() << "Construct KNNG" << std::endl;
      comm.cout0() << "========================================" << std::endl;

      recorder.start("KNNG construction");
      auto knng =
          kernel.construct(opt.k, opt.rho, opt.delta, !opt.donot_remove_dup_fvs,
                           opt.batch_size, opt.popular_fv_ratio);
      recorder.stop();

      if (opt.optimize) {
        comm.cout0() << "\n========================================"
                     << std::endl;
        comm.cout0() << "Optimize KNNG" << std::endl;
        comm.cout0() << "========================================" << std::endl;
        recorder.start("KNNG optimization");
        kernel.optimize(opt.pruning_factor, knng);
        recorder.stop();
      }

      comm.cout0() << "\n========================================" << std::endl;
      comm.cout0() << "Performance Profile" << std::endl;
      comm.cout0() << "========================================" << std::endl;
      if (opt.verbose) {
        comm.cout0() << "\nTime table (seconds):" << std::endl;
        comm.cout0() << "Name:\tMin\tMax\tMean\tStd" << std::endl;
        const auto& time_table = recorder.get_time_table();
        // For each entry, print min, mean, max, and standard deviation among
        // all MPI ranks.
        comm.cout0() << std::fixed << std::setprecision(2);
        for (const auto& entry : time_table) {
          std::vector<double> times(mpi_size);
          comm.all_gather(entry.t, times.data());
          const auto [min, mean, max, std] = utility::get_stats(times);
          for (std::size_t i = 0; i < entry.depth; ++i) {
            comm.cout0() << "  ";
          }
          comm.cout0() << entry.name << ":\t" << min << ",\t" << mean << ",\t"
                       << max << ",\t" << std << std::endl;
        }
      } else {
        const auto& time_table = recorder.get_time_table();
        for (const auto& entry : time_table) {
          if (entry.name != "KNNG construction") {
            continue;
          }
          std::vector<double> times(mpi_size);
          comm.all_gather(entry.t, times.data());
          const auto [min, mean, max, std] = utility::get_stats(times);
          comm.cout0() << "KNNG construction took (s):\t" << std::fixed
                       << std::setprecision(2) << max << std::endl;
        }
      }

      kernel.print_profile(true);

      if (!opt.knng_dump_dir.empty()) {
        comm.cout0() << "\n========================================"
                     << std::endl;
        comm.cout0() << "Dump KNNG" << std::endl;
        comm.cout0() << "========================================" << std::endl;
        comm.cout0() << "\nDump to " << opt.knng_dump_dir << std::endl;
        std::error_code ec;
        std::filesystem::create_directories(opt.knng_dump_dir, ec);
        comm.barrier();
        dump_knng(opt.knng_dump_dir + "/knng", knng, mpi_rank,
                  opt.dump_distance);
        comm.barrier();
      }
    }
    comm.barrier();
    comm.cout0() << "\n========================================" << std::endl;
    comm.cout0() << "Finished NEO DNND" << std::endl;
    comm.cout0() << "========================================" << std::endl;
  }
EXIT_NORMAL:
  ::MPI_Finalize();

  return 0;
}
// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Hnswlib benchmark program.
// Hnswlib requires data types and distance function at compile time.
// Currently, this program is designed to use the Deep1B dataset and BingANN
// dataset.

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <hnswlib/hnswlib.h>
#include "hnswlib_bench.hpp"

struct Option {
  std::string      point_file_dir;         // p
  std::string      query_file_path;        // q
  std::string      gt_file_path;           // g
  size_t           dim             = 0;    // d
  size_t           num_points      = 0;    // n
  int              M               = 48;   // m
  int              ef_construction = 200;  // c
  int              search_K        = 10;   // k
  std::vector<int> ef_list{20};            // e
};

bool parseArugments(int argc, char **argv, Option &option) {
  int n;
  while ((n = getopt(argc, argv, "p:d:n:m:c:k:e:q:g:")) != -1) {
    switch (n) {
      case 'p':
        option.point_file_dir = optarg;
        break;
      case 'd':
        option.dim = std::stoull(optarg);
        break;
      case 'n':
        option.num_points = std::stoull(optarg);
        break;
      case 'm':
        option.M = std::stoi(optarg);
        break;
      case 'c':
        option.ef_construction = std::stoi(optarg);
        break;
      case 'k':
        option.search_K = std::stoi(optarg);
        break;
      case 'e': {
        const auto ef_str_list = split(optarg, ':');
        option.ef_list.clear();
        std::transform(ef_str_list.begin(), ef_str_list.end(),
                       std::back_inserter(option.ef_list),
                       [](const std::string &s) { return std::stoi(s); });
        break;
      }
      case 'q':
        option.query_file_path = optarg;
        break;
      case 'g':
        option.gt_file_path = optarg;
        break;
      default:
        std::cerr << "Invalid argument" << std::endl;
        std::abort();
    }
  }

  return true;
}

void showOptions(const Option &opt) {
  std::cout << "point_file_dir = " << opt.point_file_dir << std::endl;
  std::cout << "dim = " << opt.dim << std::endl;
  std::cout << "num_points = " << opt.num_points << std::endl;
  std::cout << "M = " << opt.M << std::endl;
  std::cout << "ef_construction = " << opt.ef_construction << std::endl;
  std::cout << "query_file_path = " << opt.query_file_path << std::endl;
  std::cout << "search_K = " << opt.search_K << std::endl;
  std::cout << "ef = ";
  for (const auto ef : opt.ef_list) {
    std::cout << ef << " ";
  }
  std::cout << std::endl;
  std::cout << "gt_file_path = " << opt.gt_file_path << std::endl;
}

int main(int argc, char **argv) {
  Option opt;
  parseArugments(argc, argv, opt);
  showOptions(opt);

  int num_threads = 0;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    num_threads = omp_get_num_threads();
  }
  std::cout << "Number of threads = " << num_threads << std::endl;
#else
  std::cout << "OpenMP is not available" << std::endl;
#endif

  HnswSpace space(opt.dim);
  auto      alg_hnsw = hnswlib::HierarchicalNSW<DistanceT>(
      &space, opt.num_points, opt.M, opt.ef_construction,
      std::random_device()());

  std::cout << "\nStart kNN Construction" << std::endl;
  {
    const auto fpaths = findFilePaths(opt.point_file_dir);
    std::cout << "Found " << fpaths.size() << " files" << std::endl;

    const size_t batch_size  = std::max(size_t(fpaths.size() / 10), size_t(10));
    const size_t num_batches = (fpaths.size() + batch_size - 1) / batch_size;
    PointStore   points;
    double       total_time = 0;
    for (size_t batch_no = 0; batch_no < num_batches; ++batch_no) {
      std::cout << "\nBatch_no = " << batch_no << std::endl;

      std::cout << "Read points" << std::endl;
      const size_t s = batch_no * batch_size;
      const size_t n = std::min(batch_size, fpaths.size() - s);
      readPoints(
          std::vector<std::string>{fpaths.begin() + s, fpaths.begin() + s + n},
          opt.dim, points);

      std::cout << "Add points to index" << std::endl;
      const auto timer = startTimer();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < points.size(); i++) {
        alg_hnsw.addPoint(points[i].second.data(), points[i].first);
      }
      const auto tsec = geElapsedTime(timer);
      total_time += tsec;
    }
    std::cout << "k-NNG construction time (s)\t" << total_time << std::endl;
  }

  std::cout << "\nQuery" << std::endl;
  std::vector<std::vector<PointT>> queries;
  readQueries(opt.query_file_path, opt.dim, queries);
  for (const auto ef : opt.ef_list) {
    alg_hnsw.setEf(ef);
    std::cout << "ef = " << ef << std::endl;
    const auto      timer = startTimer();
    NeighborsTableT results(queries.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < queries.size(); i++) {
      auto ret = alg_hnsw.searchKnn(queries[i].data(), opt.search_K);
      // Etraxt results from priority queue
      results[i] = std::move(getContainer(ret));
    }
    const auto tsec = geElapsedTime(timer);
    std::cout << "k-NNG search time (s)\t" << tsec << std::endl;

    // Recall scores
    if (!opt.gt_file_path.empty()) {
      NeighborsTableT gt;
      readGroundTruth(opt.gt_file_path, queries.size(), opt.search_K, gt);

      {
        auto scores = getExactRecallScores(results, gt, opt.search_K);
        std::cout << "Exact recall scores (min mean max): "
                  << *std::min_element(scores.begin(), scores.end()) << "\t"
                  << std::accumulate(scores.begin(), scores.end(), 0.0) /
                         scores.size()
                  << "\t" << *std::max_element(scores.begin(), scores.end())
                  << std::endl;
      }
      {
        auto scores = getExactRecallScoresWithDistanceTies(results, gt, opt.search_K);
        std::cout << "Recall scores with distance ties (min mean max): "
                  << *std::min_element(scores.begin(), scores.end()) << "\t"
                  << std::accumulate(scores.begin(), scores.end(), 0.0) /
                         scores.size()
                  << "\t" << *std::max_element(scores.begin(), scores.end())
                  << std::endl;
      }
      std::cout << std::endl;
    }
  }

  return 0;
}

// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <saltatlas/neo_dnnd/utility.hpp>

int main() {
  // Test get_stats
  {
    std::vector<int> v{1, 2, 3, 4, 5};
    const auto [min, max, mean, var] =
        saltatlas::neo_dnnd::utility::get_stats(v);
    assert(min == 1);
    assert(max == 5);
    assert(mean == 3.0);
    assert(std::abs(var - 1.4142) < 1e-4);
  }

  // Test gen_round_robin_tournament
  {
    for (std::size_t num_players = 2; num_players < (1 << 10);
         num_players *= 2) {
      std::vector<std::vector<std::size_t>> schedule_table;
      const std::size_t                     num_arounds = num_players - 1;
      for (int player_id = 0; player_id < num_players; ++player_id) {
        std::vector<std::size_t> opponents(num_arounds);
        saltatlas::neo_dnnd::utility::gen_round_robin_tournament(
            num_players, player_id, opponents.begin());
        assert(opponents.size() == num_arounds);
        schedule_table.push_back(opponents);
      }

      // Make sure 1) each player plays with every other player exactly once and
      // 2) each player plays with only one player in each round.
      for (int i = 0; i < num_arounds; ++i) {
        // Construct a check table
        std::unordered_map<int, int> check_table;
        for (int p = 0; p < num_players; ++p) {
          const int opponent = schedule_table[p][i];
          assert(opponent != p);
          assert(check_table.find(p) == check_table.end());
          check_table[p] = opponent;
        }

        // Check the size
        assert(check_table.size() == num_players);

        // Check the reverse
        for (const auto& [p, o] : check_table) {
          assert(check_table.find(o) != check_table.end());
          assert(check_table.at(o) == p);
        }
      }
    }
  }

  std::cout << "All tests passed" << std::endl;
  return 0;
}
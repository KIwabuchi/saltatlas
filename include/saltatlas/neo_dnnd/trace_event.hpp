// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <cassert>
#include <stack>
#include <string>
#include <vector>
#include <iostream>

#include <boost/json/src.hpp>

#include "time.hpp"

namespace saltatlas::neo_dnnd {
class complete_event_recorder : public time_recorder_base {
 public:
  complete_event_recorder(const int pid, const int tid = 0)
      : m_launch_time(std::chrono::high_resolution_clock::now()),
        m_pid(pid),
        m_tid(tid) {}

  void turn_on() override { m_record = true; }

  void turn_off() override { m_record = false; }

  void reset() override {
    m_launch_time = std::chrono::high_resolution_clock::now();
    m_events.clear();
    while (!m_ts_stack.empty()) m_ts_stack.pop();
    while (!m_name_stack.empty()) m_name_stack.pop();
  }

  void start(const std::string& name) override {
    if (!m_record) return;
    const std::string& category = "default";
    m_name_stack.push(std::make_pair(name, category));
    m_ts_stack.push(get_elapsed_us(m_launch_time));
  }

  double stop() override {
    if (!m_record) return 0.0;

    assert(m_ts_stack.size() > 0);
    assert(m_name_stack.size() > 0);

    const auto start_time = m_ts_stack.top();
    m_ts_stack.pop();
    const auto duration_time = get_elapsed_us(m_launch_time) - start_time;

    const auto name = m_name_stack.top();
    m_name_stack.pop();

    boost::json::object event;
    event["name"] = name.first;
    event["cat"] = name.second;
    event["pid"] = m_pid;
    event["tid"] = m_tid;
    event["ph"] = "X";
    event["ts"] = start_time;
    event["dur"] = duration_time;
    m_events.emplace_back(boost::json::serialize(event));

    return (double)duration_time / 1e6;
  }

  const std::vector<std::string>& events() const { return m_events; }

  const std::string& get_current_name() const { return m_name_stack.top().first; }

 private:
  std::chrono::high_resolution_clock::time_point m_launch_time;
  int m_pid;
  int m_tid;
  std::stack<uint64_t> m_ts_stack{};
  std::stack<std::pair<std::string, std::string>> m_name_stack{};
  std::vector<std::string> m_events{};
  bool m_record{true};
};
}  // namespace saltatlas::neo_dnnd
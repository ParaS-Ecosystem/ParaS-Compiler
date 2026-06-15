/**
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 *
 * This file is part of the ParaS Compiler, a component of the ParaS Ecosystem.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_EVENT_HPP__
#define __PARAS_EVENT_HPP__

#include <memory>
#include <thread>
#include <vector>

namespace sycl {

class event {
private:
  std::shared_ptr<void> m_impl;
  std::vector<event> m_deps;
  bool m_ready = true;
  bool m_profile_enabled = false;
  uint64_t m_submit = 0, m_start = 0, m_end = 0;

  void wait_deps() const {
    for (auto &d : m_deps)
      d.wait();
  }

public:
  event() = default;
  ~event() = default;
  event(const event &) = default;
  event(event &&) = default;
  event &operator=(const event &) = default;
  event &operator=(event &&) = default;

  void set_dependencies(const std::vector<event> &deps) { m_deps = deps; }

  void wait() const {
    wait_deps();
    while (!m_ready)
      std::this_thread::yield();
  }

  void wait_and_throw() const { wait(); }

  std::vector<event> get_wait_list() { return m_deps; }

  static void wait(const std::vector<event> &events) {
    for (const auto &e : events)
      e.wait();
  }

  static void wait_and_throw(const std::vector<event> &events) {
    for (const auto &e : events)
      e.wait_and_throw();
  }

  template <typename Param> typename Param::return_type get_info() const {
    return typename Param::return_type();
  }

  template <typename Param>
  typename Param::return_type get_backend_info() const {
    return typename Param::return_type();
  }

  template <typename Param>
  typename Param::return_type get_profiling_info() const {
    if (!m_profile_enabled)
      throw std::runtime_error("Profiling not enabled");
    while (!m_ready)
      std::this_thread::yield();
    return typename Param::return_type();
  }

  void set_profiling(bool enabled) { m_profile_enabled = enabled; }
  void mark_complete() { m_ready = true; }
  bool is_ready() const { return m_ready; }
};

} // namespace sycl

#endif

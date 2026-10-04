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
#include <type_traits>
#include <stdexcept>
#include <mutex>
#include <future>
#include <exception>
#include <chrono>
#include "backend.hpp"
#include <thread>
#include <vector>

namespace sycl {
class event;
namespace detail {

struct event_state {
  std::shared_future<void> done;
  std::vector<event> deps;
  std::exception_ptr error;
  bool profiling = false;
};
}

class event {
  std::shared_ptr<detail::event_state> m_state =
      std::make_shared<detail::event_state>();

public:
  event() = default;
  ~event() = default;
  event(const event &) = default;
  event(event &&) = default;
  event &operator=(const event &) = default;
  event &operator=(event &&) = default;

  static event paras_async(std::shared_future<void> done,
                           std::vector<event> deps) {
    event e;
    e.m_state->done = std::move(done);
    e.m_state->deps = std::move(deps);
    return e;
  }
  std::shared_ptr<detail::event_state> paras_state() const { return m_state; }
  bool paras_complete() const {
    return !m_state->done.valid() ||
           m_state->done.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready;
  }

  void set_dependencies(const std::vector<event> &deps) {
    m_state->deps = deps;
  }

  friend bool operator==(const event &a, const event &b) {
    return a.m_state == b.m_state;
  }
  friend bool operator!=(const event &a, const event &b) { return !(a == b); }
  const void *paras_identity() const { return m_state.get(); }

  backend get_backend() const noexcept { return backend::host; }

  void wait() const {
    if (m_state->done.valid())
      m_state->done.wait();
  }

  void wait_and_throw() const { wait(); }

  std::vector<event> get_wait_list() { return m_state->deps; }

  static void wait(const std::vector<event> &events) {
    for (const auto &e : events)
      e.wait();
  }
  static void wait_and_throw(const std::vector<event> &events) {
    for (const auto &e : events)
      e.wait_and_throw();
  }

  template <typename Param> typename Param::return_type get_info() const {
    using R = typename Param::return_type;
    if constexpr (std::is_enum_v<R>)
      return static_cast<R>(paras_complete() ? 2 : 1);
    else
      return R();
  }

  template <typename Param>
  typename Param::return_type get_backend_info() const {
    return typename Param::return_type();
  }

  template <typename Param>
  typename Param::return_type get_profiling_info() const {
    if (!m_state->profiling)
      throw std::runtime_error("Profiling not enabled");
    wait();
    return typename Param::return_type();
  }

  void set_profiling(bool enabled) { m_state->profiling = enabled; }
  void mark_complete() {}
  bool is_ready() const { return paras_complete(); }
};

namespace detail {

struct pending_events {
  std::mutex mtx;
  std::vector<event> events;
  void add(const event &e) {
    if (e.paras_complete())
      return;
    std::lock_guard<std::mutex> lock(mtx);
    events.push_back(e);
  }
  void wait_all() {
    std::vector<event> todo;
    {
      std::lock_guard<std::mutex> lock(mtx);
      todo.swap(events);
    }
    for (auto &e : todo)
      e.wait();
  }
};
}

}

#endif

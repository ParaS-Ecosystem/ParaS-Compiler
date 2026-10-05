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

#ifndef SYCL_EVENT_HPP
#define SYCL_EVENT_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include "device.hpp"
#include "exception.hpp"
#include "info.hpp"
#include <chrono>
#include <cstdint>
#include <type_traits>

namespace sycl {

struct paras_event_state {
    bool profiling = false;
    std::uint64_t submit = 0, start = 0, end = 0;
};

inline std::uint64_t paras_now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

class event {
private:
    std::shared_ptr<paras_event_state> m_impl = std::make_shared<paras_event_state>();
    std::vector<event> m_deps;
    bool m_ready = true;
    backend backend_ = backend::host;

    void wait_deps() const {
        for (auto& d : m_deps)
            d.wait();
    }

public:
    event() = default;
    explicit event(backend event_backend) : backend_(event_backend) {}
    ~event() = default;
    event(const event&) = default;
    event(event&&) = default;
    event& operator=(const event&) = default;
    event& operator=(event&&) = default;

    bool operator==(const event& rhs) const noexcept { return m_impl == rhs.m_impl; }

    bool operator!=(const event& rhs) const noexcept { return !(*this == rhs); }

    backend get_backend() const noexcept { return backend_; }

    void set_dependencies(const std::vector<event>& deps) { m_deps = deps; }

    void wait() const { wait_deps(); }

    void wait_and_throw() const { wait(); }

    std::vector<event> get_wait_list() const { return m_deps; }

    static void wait(const std::vector<event>& events) {
        for (const auto& e : events)
            e.wait();
    }

    static void wait_and_throw(const std::vector<event>& events) {
        for (const auto& e : events)
            e.wait_and_throw();
    }

    template <typename Param>
    typename Param::return_type get_info() const {
        if constexpr (std::is_same_v<Param, info::event::command_execution_status>) {
            return info::event_command_status::complete;
        } else {
            return typename Param::return_type();
        }
    }

    template <typename Param>
    typename Param::return_type get_backend_info() const {
        return typename Param::return_type();
    }

    template <typename Param>
    typename Param::return_type get_profiling_info() const {
        if (!m_impl->profiling) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "event profiling info requested but the queue "
                                  "was not created with property::queue::enable_profiling");
        }
        wait();
        if constexpr (std::is_same_v<Param, info::event_profiling::command_submit>) {
            return m_impl->submit;
        } else if constexpr (std::is_same_v<Param, info::event_profiling::command_start>) {
            return m_impl->start;
        } else {
            return m_impl->end;
        }
    }

    static event paras_profiled(std::uint64_t submit, std::uint64_t start, std::uint64_t end) {
        event e;
        e.m_impl->profiling = true;
        e.m_impl->submit = submit;
        e.m_impl->start = start;
        e.m_impl->end = end;
        return e;
    }

    const void* identity() const noexcept { return m_impl.get(); }

    void set_profiling(bool enabled) { m_impl->profiling = enabled; }
    void mark_complete() { m_ready = true; }
    bool is_ready() const { return m_ready; }
};

} // namespace sycl

namespace std {
template <>
struct hash<sycl::event> {
    std::size_t operator()(const sycl::event& e) const noexcept {
        return std::hash<const void*>{}(e.identity());
    }
};
} // namespace std

#endif

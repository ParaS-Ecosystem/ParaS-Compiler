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
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace sycl {

using paras_async_handler_fn = std::function<void(sycl::exception_list)>;

struct paras_event_state {
    bool profiling = false;
    std::uint64_t submit = 0, start = 0, end = 0;

    std::mutex paras_mtx;
    std::condition_variable paras_cv;
    bool paras_complete = true;
    std::exception_ptr paras_async_error;
    bool paras_error_consumed = false;
    std::shared_ptr<paras_async_handler_fn> paras_handler;

    void paras_wait_complete() {
        std::unique_lock<std::mutex> lk(paras_mtx);
        paras_cv.wait(lk, [this] { return paras_complete; });
    }
    bool paras_is_complete() {
        std::lock_guard<std::mutex> lk(paras_mtx);
        return paras_complete;
    }
};

inline thread_local bool paras_in_host_task_worker = false;

struct paras_host_task_registry {
    std::mutex mtx;
    std::vector<std::shared_ptr<paras_event_state>> states;

    std::vector<std::shared_ptr<paras_event_state>> snapshot() {
        std::lock_guard<std::mutex> lk(mtx);
        return states;
    }
    void add(std::shared_ptr<paras_event_state> s) {
        std::lock_guard<std::mutex> lk(mtx);
        states.push_back(std::move(s));
    }
    void prune() {
        std::lock_guard<std::mutex> lk(mtx);
        std::vector<std::shared_ptr<paras_event_state>> keep;
        for (auto& s : states) {
            std::lock_guard<std::mutex> slk(s->paras_mtx);
            if (!s->paras_complete || (s->paras_async_error && !s->paras_error_consumed))
                keep.push_back(s);
        }
        states.swap(keep);
    }
    ~paras_host_task_registry() {
        for (auto& s : snapshot())
            s->paras_wait_complete();
    }
};

inline paras_host_task_registry& paras_host_tasks() {
    static paras_host_task_registry r;
    return r;
}

inline void paras_drain_host_tasks() {
    if (paras_in_host_task_worker)
        return;
    for (auto& s : paras_host_tasks().snapshot())
        s->paras_wait_complete();
    paras_host_tasks().prune();
}

inline void paras_report_async_errors(const std::vector<std::shared_ptr<paras_event_state>>& sts) {
    std::vector<std::pair<std::shared_ptr<paras_async_handler_fn>, sycl::exception_list>> groups;
    for (auto& s : sts) {
        std::exception_ptr err;
        {
            std::lock_guard<std::mutex> lk(s->paras_mtx);
            if (!s->paras_async_error || s->paras_error_consumed)
                continue;
            s->paras_error_consumed = true;
            err = s->paras_async_error;
        }
        auto it = groups.begin();
        for (; it != groups.end(); ++it)
            if (it->first == s->paras_handler)
                break;
        if (it == groups.end()) {
            groups.emplace_back(s->paras_handler, sycl::exception_list{});
            it = std::prev(groups.end());
        }
        it->second.paras_push_back(err);
    }
    for (auto& [h, list] : groups) {
        if (h && *h) {
            (*h)(list);
        } else {
            std::fprintf(stderr, "ParaS: unhandled asynchronous SYCL error and no "
                                 "async_handler was provided; terminating\n");
            std::terminate();
        }
    }
    paras_host_tasks().prune();
}

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

    void wait() const {
        wait_deps();
        m_impl->paras_wait_complete();
    }

    void wait_and_throw() const {
        wait();
        std::vector<std::shared_ptr<paras_event_state>> related;
        std::unordered_set<const paras_event_state*> seen;
        std::vector<const event*> todo{this};
        while (!todo.empty()) {
            const event* e = todo.back();
            todo.pop_back();
            if (!seen.insert(e->m_impl.get()).second)
                continue;
            related.push_back(e->m_impl);
            for (const auto& d : e->m_deps)
                todo.push_back(&d);
        }
        paras_report_async_errors(related);
    }

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
            if (!m_impl->paras_is_complete())
                return info::event_command_status::running;
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

    const std::shared_ptr<paras_event_state>& paras_state() const noexcept { return m_impl; }
};

inline event paras_launch_host_task(std::function<void()> body, const std::vector<event>& deps,
                                    std::shared_ptr<paras_async_handler_fn> handler) {
    event e;
    auto st = e.paras_state();
    st->paras_complete = false;
    st->paras_handler = std::move(handler);
    e.set_dependencies(deps);

    auto prior = paras_host_tasks().snapshot();
    paras_host_tasks().prune();
    paras_host_tasks().add(st);

    std::thread([st, deps, prior, body = std::move(body)]() {
        paras_in_host_task_worker = true;
        for (const auto& d : deps)
            d.wait();
        for (const auto& p : prior)
            p->paras_wait_complete();
        std::exception_ptr err;
        try {
            body();
        } catch (...) {
            err = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lk(st->paras_mtx);
            st->paras_async_error = err;
            st->paras_complete = true;
        }
        st->paras_cv.notify_all();
    }).detach();
    return e;
}

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

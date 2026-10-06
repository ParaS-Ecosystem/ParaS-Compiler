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

#ifndef __PARAS_HOST_GROUP_RUNTIME_HPP__
#define __PARAS_HOST_GROUP_RUNTIME_HPP__

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace paras_host_detail {

class group_barrier_t {
public:
    explicit group_barrier_t(std::size_t count) : count_(count) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(m_);
        const std::size_t gen = generation_;
        if (++waiting_ == count_) {
            waiting_ = 0;
            ++generation_;
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return generation_ != gen; });
        }
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::size_t count_;
    std::size_t waiting_ = 0;
    std::size_t generation_ = 0;
};

inline thread_local group_barrier_t* tl_barrier = nullptr;
inline thread_local unsigned char* tl_local_mem = nullptr;
inline thread_local unsigned char* tl_group_scratch = nullptr;
inline thread_local std::size_t tl_local_linear_id = 0;
inline thread_local std::size_t tl_local_size = 1;

inline void work_group_barrier() {
    if (tl_barrier)
        tl_barrier->arrive_and_wait();
}

constexpr std::size_t kHostLocalMemBytes = 65536;

template <typename RunItem>
void run_work_groups(std::size_t num_groups, std::size_t local_size, RunItem run_item) {
    if (num_groups == 0 || local_size == 0)
        return;

    std::size_t hw = std::thread::hardware_concurrency();
    if (const char* e = std::getenv("PARAS_HOST_THREADS")) {
        const long v = std::atol(e);
        if (v > 0)
            hw = static_cast<std::size_t>(v);
    }
    hw = std::max<std::size_t>(1, hw);
    const std::size_t teams =
        std::max<std::size_t>(1, std::min(num_groups, (hw + local_size - 1) / local_size));

    std::atomic<std::size_t> next_group{0};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    struct team_state {
        group_barrier_t barrier;
        std::unique_ptr<unsigned char[]> local_mem;
        std::unique_ptr<unsigned char[]> group_scratch;
        std::size_t current_group = 0;
        bool abort = false;
        explicit team_state(std::size_t L)
            : barrier(L), local_mem(new unsigned char[kHostLocalMemBytes]),
              group_scratch(new unsigned char[kHostLocalMemBytes]) {}
    };
    std::vector<std::unique_ptr<team_state>> states;
    for (std::size_t t = 0; t < teams; ++t)
        states.emplace_back(new team_state(local_size));

    std::vector<std::thread> threads;
    threads.reserve(teams * local_size);

    for (std::size_t t = 0; t < teams; ++t) {
        team_state* ts = states[t].get();
        for (std::size_t lid = 0; lid < local_size; ++lid) {
            threads.emplace_back([&, ts, lid]() {
                tl_barrier = &ts->barrier;
                tl_local_mem = ts->local_mem.get();
                tl_group_scratch = ts->group_scratch.get();
                tl_local_linear_id = lid;
                tl_local_size = local_size;
                for (;;) {
                    if (lid == 0) {
                        ts->current_group = next_group.fetch_add(1);
                        std::lock_guard<std::mutex> g(error_mutex);
                        if (first_error)
                            ts->abort = true;
                    }
                    ts->barrier.arrive_and_wait();
                    const std::size_t grp = ts->current_group;
                    const bool stop = ts->abort || grp >= num_groups;
                    ts->barrier.arrive_and_wait();
                    if (stop)
                        break;

                    try {
                        run_item(grp, lid);
                    } catch (...) {
                        std::lock_guard<std::mutex> g(error_mutex);
                        if (!first_error)
                            first_error = std::current_exception();
                    }
                    ts->barrier.arrive_and_wait();
                }
                tl_barrier = nullptr;
                tl_local_mem = nullptr;
                tl_group_scratch = nullptr;
                tl_local_size = 1;
            });
        }
    }
    for (auto& th : threads)
        th.join();
    if (first_error)
        std::rethrow_exception(first_error);
}

} // namespace paras_host_detail

#endif // !device pass
#endif // __PARAS_HOST_GROUP_RUNTIME_HPP__

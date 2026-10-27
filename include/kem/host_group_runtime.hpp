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

#include "kem/host_pool.hpp"

namespace paras_host_detail {

class group_barrier_t {
public:
    explicit group_barrier_t(std::size_t count, long spin_us = 0)
        : count_(count), spin_us_(spin_us) {}
    std::size_t size() const noexcept { return count_; }

    void arrive_and_wait() {
        const unsigned phase = phase_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            arrived_.store(0, std::memory_order_relaxed);
            phase_.fetch_add(1, std::memory_order_seq_cst);
            if (sleepers_.load(std::memory_order_seq_cst)) {
                { std::lock_guard<std::mutex> lk(m_); }
                cv_.notify_all();
            }
            return;
        }
        auto passed = [&] { return phase_.load(std::memory_order_acquire) != phase; };
        if (spin_until(passed, spin_us_)) return;
        std::unique_lock<std::mutex> lk(m_);
        sleepers_.fetch_add(1, std::memory_order_seq_cst);
        cv_.wait(lk, [&] {
            return phase_.load(std::memory_order_seq_cst) != phase;
        });
        sleepers_.fetch_sub(1, std::memory_order_relaxed);
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::size_t count_;
    long spin_us_;

    alignas(64) std::atomic<std::size_t> arrived_{0};
    alignas(64) std::atomic<unsigned> phase_{0};
    alignas(64) std::atomic<std::size_t> sleepers_{0};
};

inline thread_local group_barrier_t* tl_barrier   = nullptr;
inline thread_local unsigned char*    tl_local_mem = nullptr;

inline thread_local unsigned char*    tl_group_scratch = nullptr;
inline thread_local std::size_t       tl_local_linear_id = 0;

inline void work_group_barrier() {
    if (tl_barrier) tl_barrier->arrive_and_wait();
}

constexpr std::size_t kHostLocalMemBytes = 65536;

template <typename RunItem>
void run_work_groups(std::size_t num_groups, std::size_t local_size,
                     RunItem run_item) {
    if (num_groups == 0 || local_size == 0) return;

    const std::size_t hw = host_core_count();

    const std::size_t teams =
        std::max<std::size_t>(1, std::min(num_groups, (hw + local_size - 1) / local_size));

    const long spin = teams * local_size <= hw ? host_spin_us() : 200;

    struct team_state {
        group_barrier_t barrier;
        std::unique_ptr<unsigned char[]> local_mem;
        std::unique_ptr<unsigned char[]> group_scratch;
        team_state(std::size_t L, long spin)
            : barrier(L, spin), local_mem(new unsigned char[kHostLocalMemBytes]),
              group_scratch(new unsigned char[kHostLocalMemBytes]) {}
    };
    std::vector<std::unique_ptr<team_state>> states;
    for (std::size_t t = 0; t < teams; ++t)
        states.emplace_back(new team_state(local_size, spin));

    std::exception_ptr first_error;
    std::mutex error_mutex;

    host_pool::instance().run(teams * local_size, [&](std::size_t slot) {
        const std::size_t t = slot / local_size;
        const std::size_t lid = slot % local_size;
        team_state* ts = states[t].get();
        tl_barrier   = &ts->barrier;
        tl_local_mem = ts->local_mem.get();
        tl_group_scratch   = ts->group_scratch.get();
        tl_local_linear_id = lid;
        for (std::size_t grp = t; grp < num_groups; grp += teams) {
            try {
                run_item(grp, lid);
            } catch (...) {
                std::lock_guard<std::mutex> g(error_mutex);
                if (!first_error) first_error = std::current_exception();
            }

            ts->barrier.arrive_and_wait();
        }
        tl_barrier   = nullptr;
        tl_local_mem = nullptr;
        tl_group_scratch = nullptr;
    });
    if (first_error) std::rethrow_exception(first_error);
}

}

#endif
#endif

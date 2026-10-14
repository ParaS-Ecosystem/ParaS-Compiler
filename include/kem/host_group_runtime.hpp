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
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <unistd.h>

namespace paras_host_detail {

inline std::size_t host_hardware_threads() {
    static const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    return hw;
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

class group_barrier_t {
public:
    explicit group_barrier_t(std::size_t count)
        : count_(count), spins_(count <= host_hardware_threads() ? 4000 : 0) {}

    void arrive_and_wait() {
        const std::size_t gen = generation_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            arrived_.store(0, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(m_);
                generation_.store(gen + 1, std::memory_order_release);
            }
            cv_.notify_all();
            return;
        }
        for (int i = 0; i < spins_; ++i) {
            if (generation_.load(std::memory_order_acquire) != gen)
                return;
            cpu_relax();
        }
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != gen; });
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::size_t count_;
    int spins_;
    std::atomic<std::size_t> arrived_{0};
    std::atomic<std::size_t> generation_{0};
};

class host_worker_pool {
public:
    static host_worker_pool& instance() {
        static host_worker_pool* pool = new host_worker_pool(); // never destroyed
        return *pool;
    }

    bool try_run(std::size_t n, const std::function<void(std::size_t)>& fn) {
        if (in_worker() || ::getpid() != owner_pid_)
            return false;
        std::unique_lock<std::mutex> busy(busy_, std::try_to_lock);
        if (!busy.owns_lock())
            return false;

        while (slots_.size() < n)
            add_worker();

        job_ = &fn;
        remaining_.store(n, std::memory_order_release);
        ++job_gen_;
        for (std::size_t i = 0; i < n; ++i) {
            slot& s = *slots_[i];
            {
                std::lock_guard<std::mutex> lk(s.m);
                s.gen = job_gen_;
            }
            s.cv.notify_one();
        }

        std::unique_lock<std::mutex> lk(done_m_);
        done_cv_.wait(lk, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
        job_ = nullptr;
        return true;
    }

private:
    struct slot {
        std::mutex m;
        std::condition_variable cv;
        std::size_t gen = 0;
    };

    host_worker_pool() : owner_pid_(::getpid()) {}

    static bool& in_worker() {
        static thread_local bool flag = false;
        return flag;
    }

    void add_worker() {
        const std::size_t idx = slots_.size();
        slots_.emplace_back(new slot());
        slot* s = slots_.back().get();
        std::thread([this, s, idx]() {
            in_worker() = true;
            std::size_t seen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(s->m);
                    s->cv.wait(lk, [&] { return s->gen != seen; });
                    seen = s->gen;
                }
                try {
                    (*job_)(idx);
                } catch (...) {
                }
                if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(done_m_);
                    done_cv_.notify_all();
                }
            }
        }).detach();
    }

    const pid_t owner_pid_;
    std::mutex busy_;
    std::vector<std::unique_ptr<slot>> slots_;
    const std::function<void(std::size_t)>* job_ = nullptr;
    std::size_t job_gen_ = 0;
    std::atomic<std::size_t> remaining_{0};
    std::mutex done_m_;
    std::condition_variable done_cv_;
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

inline unsigned char* host_local_scratch() {
    static thread_local std::unique_ptr<unsigned char[]> scratch;
    if (!scratch)
        scratch.reset(new unsigned char[kHostLocalMemBytes]());
    return scratch.get();
}

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

    const std::function<void(std::size_t)> work_item_thread = [&](std::size_t index) {
        team_state* ts = states[index / local_size].get();
        const std::size_t lid = index % local_size;
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
        tl_local_linear_id = 0;
        tl_local_size = 1;
    };

    const std::size_t total = teams * local_size;
    if (!host_worker_pool::instance().try_run(total, work_item_thread)) {
        std::vector<std::thread> threads;
        threads.reserve(total);
        for (std::size_t i = 0; i < total; ++i)
            threads.emplace_back(work_item_thread, i);
        for (auto& th : threads)
            th.join();
    }
    if (first_error)
        std::rethrow_exception(first_error);
}

} // namespace paras_host_detail

#endif // !device pass
#endif // __PARAS_HOST_GROUP_RUNTIME_HPP__

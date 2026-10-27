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
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "kem/host_pool.hpp"

#if defined(__x86_64__) && defined(__linux__) && !defined(PARAS_HOST_NO_FIBERS)
#define PARAS_HOST_FIBERS 1
#include <sys/mman.h>

extern "C" void paras_fiber_switch(void **save, void *to);
extern "C" void paras_fiber_entry();
__asm__(".text\n"
        ".weak paras_fiber_switch\n"
        ".type paras_fiber_switch,@function\n"
        "paras_fiber_switch:\n"
        "  .byte 0xf3,0x0f,0x1e,0xfa\n"
        "  pushq %rbp\n  pushq %rbx\n  pushq %r12\n"
        "  pushq %r13\n  pushq %r14\n  pushq %r15\n"
        "  subq $8, %rsp\n"
        "  movq %rsp, (%rdi)\n"
        "  movq %rsi, %rsp\n"
        "  addq $8, %rsp\n"
        "  popq %r15\n  popq %r14\n  popq %r13\n"
        "  popq %r12\n  popq %rbx\n  popq %rbp\n"
        "  ret\n"
        ".size paras_fiber_switch, .-paras_fiber_switch\n"
        ".weak paras_fiber_entry\n"
        ".type paras_fiber_entry,@function\n"
        "paras_fiber_entry:\n"
        "  .byte 0xf3,0x0f,0x1e,0xfa\n"
        "  movq %r12, %rdi\n"
        "  callq *%r13\n"
        "  ud2\n"
        ".size paras_fiber_entry, .-paras_fiber_entry\n");
#endif

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

#if PARAS_HOST_FIBERS

struct fiber_team {
    struct item {
        void* sp = nullptr;
        bool done = true;
    };
    std::size_t L = 0, cur = 0, grp = 0;
    bool direct = false;
    std::vector<item> items;
    void* sched_sp = nullptr;
    void (*run)(void*, std::size_t, std::size_t) = nullptr;
    void* run_ctx = nullptr;
    std::exception_ptr* err = nullptr;
    std::mutex* err_m = nullptr;

    void yield() {
        if (!direct) paras_fiber_switch(&items[cur].sp, sched_sp);
    }
};
inline thread_local fiber_team* tl_fibers = nullptr;
#endif

inline thread_local group_barrier_t* tl_barrier   = nullptr;
inline thread_local unsigned char*    tl_local_mem = nullptr;

inline thread_local unsigned char*    tl_group_scratch = nullptr;
inline thread_local std::size_t       tl_local_linear_id = 0;

inline void work_group_barrier() {
#if PARAS_HOST_FIBERS
    if (tl_fibers) { tl_fibers->yield(); return; }
#endif
    if (tl_barrier) tl_barrier->arrive_and_wait();
}

constexpr std::size_t kHostLocalMemBytes = 65536;

#if PARAS_HOST_FIBERS
inline bool host_fibers_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("PARAS_HOST_FIBERS");
        return !(e && e[0] == '0');
    }();
    return on;
}
inline std::size_t host_fiber_stack_bytes() {
    static const std::size_t b = [] {
        std::size_t kb = 256;
        if (const char* e = std::getenv("PARAS_HOST_FIBER_STACK_KB")) {
            const long v = std::atol(e);
            if (v >= 16) kb = static_cast<std::size_t>(v);
        }
        return kb * 1024;
    }();
    return b;
}

struct fiber_stacks {
    unsigned char* base = nullptr;
    std::size_t count = 0, bytes = 0;
    static constexpr std::size_t kGuard = 4096;
    ~fiber_stacks() { release(); }
    void release() {
        if (base) munmap(base, kGuard + count * bytes);
        base = nullptr;
        count = 0;
    }

    unsigned char* top(std::size_t i) { return base + kGuard + (i + 1) * bytes; }
    bool reserve(std::size_t n) {
        const std::size_t want = host_fiber_stack_bytes();
        if (base && count >= n && bytes == want) return true;
        release();
        void* m = mmap(nullptr, kGuard + n * want, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) return false;
        mprotect(m, kGuard, PROT_NONE);
        base = static_cast<unsigned char*>(m);
        count = n;
        bytes = want;
        return true;
    }
};
inline fiber_stacks& my_fiber_stacks() {
    static thread_local fiber_stacks s;
    return s;
}

inline void fiber_main(void* arg) {
    fiber_team* t = static_cast<fiber_team*>(arg);
    const std::size_t lid = t->cur;
    try {
        t->run(t->run_ctx, t->grp, lid);
    } catch (...) {
        std::lock_guard<std::mutex> g(*t->err_m);
        if (!*t->err) *t->err = std::current_exception();
    }
    t->items[lid].done = true;
    paras_fiber_switch(&t->items[lid].sp, t->sched_sp);
    __builtin_trap();
}

inline void* fiber_init(unsigned char* top, fiber_team* t) {
    void** s = reinterpret_cast<void**>(top - 80);
    s[0] = nullptr;
    s[1] = nullptr;
    s[2] = nullptr;
    s[3] = reinterpret_cast<void*>(&fiber_main);
    s[4] = t;
    s[5] = nullptr;
    s[6] = nullptr;
    s[7] = reinterpret_cast<void*>(&paras_fiber_entry);
    return s;
}

template <typename RunItem>
bool run_work_groups_fibers(std::size_t num_groups, std::size_t local_size,
                            RunItem& run_item) {
    const std::size_t hw = host_core_count();
    const std::size_t workers = std::max<std::size_t>(1, std::min(num_groups, hw));
    std::atomic<std::size_t> next{0};
    std::atomic<bool> no_stack{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    host_pool::instance().run(workers, [&](std::size_t) {
        fiber_stacks& st = my_fiber_stacks();
        if (local_size > 1 && !st.reserve(local_size)) {
            no_stack = true;
            return;
        }
        thread_local std::unique_ptr<unsigned char[]> local_mem, scratch;
        if (!local_mem) {
            local_mem.reset(new unsigned char[kHostLocalMemBytes]);
            scratch.reset(new unsigned char[kHostLocalMemBytes]);
        }
        group_barrier_t size_only(local_size);
        fiber_team team;
        team.L = local_size;
        team.items.resize(local_size);
        team.run = [](void* ctx, std::size_t g, std::size_t l) {
            (*static_cast<RunItem*>(ctx))(g, l);
        };
        team.run_ctx = &run_item;
        team.err = &first_error;
        team.err_m = &error_mutex;
        tl_barrier = &size_only;
        tl_local_mem = local_mem.get();
        tl_group_scratch = scratch.get();
        for (;;) {
            const std::size_t grp = next.fetch_add(1, std::memory_order_relaxed);
            if (grp >= num_groups) break;
            if (local_size == 1) {
                tl_local_linear_id = 0;
                try {
                    run_item(grp, 0);
                } catch (...) {
                    std::lock_guard<std::mutex> g(error_mutex);
                    if (!first_error) first_error = std::current_exception();
                }
                continue;
            }
            team.grp = grp;
            tl_fibers = &team;

            team.direct = false;
            team.items[0].done = false;
            team.items[0].sp = fiber_init(st.top(0), &team);
            team.cur = 0;
            tl_local_linear_id = 0;
            paras_fiber_switch(&team.sched_sp, team.items[0].sp);
            if (team.items[0].done) {
                team.direct = true;
                for (std::size_t l = 1; l < local_size; ++l) {
                    tl_local_linear_id = l;
                    try {
                        run_item(grp, l);
                    } catch (...) {
                        std::lock_guard<std::mutex> g(error_mutex);
                        if (!first_error) first_error = std::current_exception();
                    }
                }
                team.direct = false;
                tl_fibers = nullptr;
                continue;
            }

            std::size_t live = local_size;
            for (std::size_t l = 1; l < local_size; ++l) {
                team.items[l].done = false;
                team.items[l].sp = fiber_init(st.top(l), &team);
                team.cur = l;
                tl_local_linear_id = l;
                paras_fiber_switch(&team.sched_sp, team.items[l].sp);
                if (team.items[l].done) --live;
            }
            while (live) {
                for (std::size_t l = 0; l < local_size; ++l) {
                    if (team.items[l].done) continue;
                    team.cur = l;
                    tl_local_linear_id = l;
                    paras_fiber_switch(&team.sched_sp, team.items[l].sp);
                    if (team.items[l].done) --live;
                }
            }
            tl_fibers = nullptr;
        }
        tl_barrier = nullptr;
        tl_local_mem = nullptr;
        tl_group_scratch = nullptr;
    });

    if (no_stack && next.load() == 0) return false;
    if (first_error) std::rethrow_exception(first_error);
    return true;
}
#endif

template <typename RunItem>
void run_work_groups(std::size_t num_groups, std::size_t local_size,
                     RunItem run_item) {
    if (num_groups == 0 || local_size == 0) return;
#if PARAS_HOST_FIBERS
    if (host_fibers_enabled()) {
        if (run_work_groups_fibers(num_groups, local_size, run_item)) return;

    }
#endif

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

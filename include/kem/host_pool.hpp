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

#ifndef __PARAS_HOST_POOL_HPP__
#define __PARAS_HOST_POOL_HPP__

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace paras_host_detail {

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)

  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

inline unsigned host_core_count() noexcept {
  static const unsigned n = [] {
    unsigned hw = std::thread::hardware_concurrency();
    if (const char *e = std::getenv("PARAS_HOST_THREADS")) {
      const long v = std::atol(e);
      if (v > 0)
        hw = static_cast<unsigned>(v);
    }
    return hw == 0 ? 4u : hw;
  }();
  return n;
}

inline int host_spin_count() noexcept {
  static const int n = [] {
    if (const char *e = std::getenv("PARAS_HOST_SPIN")) {
      const long v = std::atol(e);
      if (v >= 0)
        return static_cast<int>(v);
    }
    return 4000;
  }();
  return n;
}

inline thread_local bool tl_in_host_pool = false;

class host_pool {
public:
  static host_pool &instance() {

    static host_pool *p = new host_pool();
    return *p;
  }

  template <typename F> void run(std::size_t slots, F &&fn) {
    if (slots == 0)
      return;
    if (tl_in_host_pool) {
      run_fresh_threads(slots, fn);
      return;
    }
    if (slots == 1) {
      in_pool_scope scope;
      fn(std::size_t(0));
      return;
    }
    post_and_wait(fn, slots,  true, slots - 1);
  }

  template <typename F>
  void run_blocks(std::size_t blocks, std::size_t threads, F &&fn) {
    if (blocks == 0)
      return;
    if (threads > blocks)
      threads = blocks;
    if (threads <= 1 || tl_in_host_pool) {
      in_pool_scope scope;
      for (std::size_t b = 0; b < blocks; ++b)
        fn(b);
      return;
    }
    post_and_wait(fn, blocks,  false, threads - 1);
  }

private:

  static constexpr unsigned kNextBits = 20, kLimitBits = 20;
  static constexpr std::uint64_t kField = (std::uint64_t(1) << 20) - 1;
  static constexpr std::uint64_t kGenMask = (std::uint64_t(1) << 24) - 1;
  static std::uint64_t gen_of(std::uint64_t v) { return v >> (kNextBits + kLimitBits); }
  static std::uint64_t limit_of(std::uint64_t v) { return (v >> kNextBits) & kField; }
  static std::uint64_t next_of(std::uint64_t v) { return v & kField; }

  struct in_pool_scope {
    bool prev = tl_in_host_pool;
    in_pool_scope() { tl_in_host_pool = true; }
    ~in_pool_scope() { tl_in_host_pool = prev; }
  };

  host_pool() = default;

  template <typename F>
  void post_and_wait(F &fn, std::size_t total, bool distinct,
                     std::size_t helpers) {

    std::lock_guard<std::mutex> one_job(submit_);
    grow(helpers);
    call_ = [](void *ctx, std::size_t i) { (*static_cast<F *>(ctx))(i); };
    ctx_ = static_cast<void *>(&fn);
    distinct_ = distinct;
    total_ = total;
    err_ = nullptr;
    done_.store(0, std::memory_order_relaxed);
    gen_ = (gen_ + 1) & kGenMask;
    const std::uint64_t start = distinct ? 1 : 0;
    ticket_.store((gen_ << (kNextBits + kLimitBits)) |
                      (std::uint64_t(total) << kNextBits) | start,
                  std::memory_order_seq_cst);

    const std::size_t sleeping = sleepers_.load(std::memory_order_seq_cst);
    if (sleeping) {
      { std::lock_guard<std::mutex> lk(m_); }
      if (helpers >= sleeping)
        cv_.notify_all();
      else
        for (std::size_t k = 0; k < helpers; ++k)
          cv_.notify_one();
    }
    {
      in_pool_scope scope;
      if (distinct) {
        execute(0);
      } else {
        std::uint64_t idx;
        while (claim(gen_, idx))
          execute(static_cast<std::size_t>(idx));
      }
    }

    const int spin = host_spin_count();
    for (int i = 0; i < spin && done_.load(std::memory_order_acquire) != total_; ++i)
      cpu_relax();
    if (done_.load(std::memory_order_acquire) != total_) {
      std::unique_lock<std::mutex> lk(m_);
      caller_waiting_.store(true, std::memory_order_seq_cst);
      done_cv_.wait(lk, [&] { return done_.load(std::memory_order_seq_cst) == total_; });
      caller_waiting_.store(false, std::memory_order_relaxed);
    }
    if (err_)
      std::rethrow_exception(err_);
  }

  bool claim(std::uint64_t gen, std::uint64_t &idx) {
    std::uint64_t v = ticket_.load(std::memory_order_acquire);
    for (;;) {
      if (gen_of(v) != gen || next_of(v) >= limit_of(v))
        return false;
      if (ticket_.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        idx = next_of(v);
        return true;
      }
    }
  }

  void execute(std::size_t i) {

    const std::size_t total = total_;
    try {
      call_(ctx_, i);
    } catch (...) {
      std::lock_guard<std::mutex> g(err_m_);
      if (!err_)
        err_ = std::current_exception();
    }
    if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == total &&
        caller_waiting_.load(std::memory_order_seq_cst)) {
      std::lock_guard<std::mutex> lk(m_);
      done_cv_.notify_all();
    }
  }

  void grow(std::size_t want) {
    while (nthreads_ < want) {
      std::thread([this] { loop(); }).detach();
      ++nthreads_;
    }
  }

  void loop() {
    tl_in_host_pool = true;

    std::uint64_t seen = ~std::uint64_t(0);
    const int spin = host_spin_count();
    for (;;) {
      std::uint64_t v = ticket_.load(std::memory_order_acquire);
      const std::uint64_t g = gen_of(v);
      if (g != seen) {
        std::uint64_t idx;
        if (claim(g, idx)) {

          const bool distinct = distinct_;
          execute(static_cast<std::size_t>(idx));
          if (distinct)
            seen = g;
          continue;
        }
        seen = g;
        continue;
      }
      for (int i = 0; i < spin; ++i) {
        if (gen_of(ticket_.load(std::memory_order_acquire)) != seen)
          break;
        cpu_relax();
      }
      if (gen_of(ticket_.load(std::memory_order_acquire)) != seen)
        continue;
      std::unique_lock<std::mutex> lk(m_);
      sleepers_.fetch_add(1, std::memory_order_seq_cst);
      cv_.wait(lk, [&] {
        return gen_of(ticket_.load(std::memory_order_seq_cst)) != seen;
      });
      sleepers_.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  template <typename F>
  static void run_fresh_threads(std::size_t slots, F &fn) {
    std::exception_ptr err;
    std::mutex em;
    std::vector<std::thread> ts;
    ts.reserve(slots);
    for (std::size_t s = 0; s < slots; ++s)
      ts.emplace_back([&, s] {
        tl_in_host_pool = true;
        try {
          fn(s);
        } catch (...) {
          std::lock_guard<std::mutex> g(em);
          if (!err)
            err = std::current_exception();
        }
      });
    for (auto &t : ts)
      t.join();
    if (err)
      std::rethrow_exception(err);
  }

  std::mutex submit_;
  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  std::atomic<std::uint64_t> ticket_{0};
  std::atomic<std::size_t> done_{0};
  std::atomic<std::size_t> sleepers_{0};
  std::atomic<bool> caller_waiting_{false};
  void (*call_)(void *, std::size_t) = nullptr;
  void *ctx_ = nullptr;
  bool distinct_ = false;
  std::size_t total_ = 0;
  std::uint64_t gen_ = 0;
  std::size_t nthreads_ = 0;
  std::mutex err_m_;
  std::exception_ptr err_;
};

template <typename F> void host_parallel_chunks(std::size_t n, F &&fn) {
  if (n == 0)
    return;
  const std::size_t nt = host_core_count();
  const std::size_t threads = n < nt ? n : nt;

  std::size_t blocks = threads * 4 < n ? threads * 4 : n;
  if (blocks > ((std::size_t(1) << 20) - 1))
    blocks = (std::size_t(1) << 20) - 1;
  const std::size_t bsize = (n + blocks - 1) / blocks;
  blocks = (n + bsize - 1) / bsize;
  host_pool::instance().run_blocks(blocks, threads, [&](std::size_t b) {
    const std::size_t s = b * bsize;
    const std::size_t e = s + bsize < n ? s + bsize : n;
    fn(s, e);
  });
}

}

#endif
#endif

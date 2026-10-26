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
    post_and_wait(fn, slots,  true);
  }

  template <typename F> void run_chunks(std::size_t chunks, F &&fn) {
    if (chunks == 0)
      return;
    if (chunks == 1 || tl_in_host_pool) {
      in_pool_scope scope;
      for (std::size_t c = 0; c < chunks; ++c)
        fn(c);
      return;
    }
    if (chunks > kMaxChunks)
      chunks = kMaxChunks;
    post_and_wait(fn, chunks,  false);
  }

  static constexpr std::size_t kMaxChunks = 4096;

private:
  struct alignas(64) claim_slot {
    std::atomic<std::uint64_t> gen{0};
  };

  struct in_pool_scope {
    bool prev = tl_in_host_pool;
    in_pool_scope() { tl_in_host_pool = true; }
    ~in_pool_scope() { tl_in_host_pool = prev; }
  };

  host_pool() : claims_(new claim_slot[kMaxChunks]) {}

  template <typename F>
  void post_and_wait(F &fn, std::size_t total, bool distinct) {
    std::lock_guard<std::mutex> one_job(submit_);
    grow(total - 1);
    const std::uint64_t g = gen_.load(std::memory_order_relaxed) + 1;

    call_.store(+[](void *ctx, std::size_t i) { (*static_cast<F *>(ctx))(i); },
                std::memory_order_relaxed);
    ctx_.store(static_cast<void *>(&fn), std::memory_order_relaxed);
    distinct_.store(distinct, std::memory_order_relaxed);
    total_.store(total, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    err_ = nullptr;
    gen_.store(g, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst)) {
      { std::lock_guard<std::mutex> lk(m_); }
      cv_.notify_all();
    }
    {
      in_pool_scope scope;
      const job j{call_.load(std::memory_order_relaxed), &fn, distinct, total};
      if (distinct)
        execute(j, 0);
      else
        work_chunks(j, g, 0);
    }

    const int spin = host_spin_count();
    for (int i = 0; i < spin && done_.load(std::memory_order_acquire) != total; ++i)
      cpu_relax();
    if (done_.load(std::memory_order_acquire) != total) {
      std::unique_lock<std::mutex> lk(m_);
      caller_waiting_.store(true, std::memory_order_seq_cst);
      done_cv_.wait(lk, [&] { return done_.load(std::memory_order_seq_cst) == total; });
      caller_waiting_.store(false, std::memory_order_relaxed);
    }
    if (err_)
      std::rethrow_exception(err_);
  }

  struct job {
    void (*call)(void *, std::size_t);
    void *ctx;
    bool distinct;
    std::size_t total;
  };

  void work_chunks(const job &j, std::uint64_t g, std::size_t rank) {
    for (std::size_t k = 0; k < j.total; ++k) {
      const std::size_t c = (rank + k) % j.total;
      std::uint64_t v = claims_[c].gen.load(std::memory_order_relaxed);
      if (v >= g)
        continue;
      if (claims_[c].gen.compare_exchange_strong(v, g, std::memory_order_acq_rel))
        execute(j, c);
    }
  }

  void execute(const job &j, std::size_t i) {
    try {
      j.call(j.ctx, i);
    } catch (...) {
      std::lock_guard<std::mutex> g(err_m_);
      if (!err_)
        err_ = std::current_exception();
    }
    if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == j.total &&
        caller_waiting_.load(std::memory_order_seq_cst)) {
      std::lock_guard<std::mutex> lk(m_);
      done_cv_.notify_all();
    }
  }

  void grow(std::size_t want) {
    const std::uint64_t now = gen_.load(std::memory_order_relaxed);
    while (nthreads_ < want) {
      const std::size_t rank = ++nthreads_;
      std::thread([this, rank, now] { loop(rank, now); }).detach();
    }
  }

  void loop(std::size_t rank, std::uint64_t seen) {
    tl_in_host_pool = true;
    const int spin = host_spin_count();
    for (;;) {
      const std::uint64_t g = gen_.load(std::memory_order_acquire);
      if (g != seen) {
        seen = g;
        const job j{call_.load(std::memory_order_relaxed),
                    ctx_.load(std::memory_order_relaxed),
                    distinct_.load(std::memory_order_relaxed),
                    total_.load(std::memory_order_relaxed)};

        if (gen_.load(std::memory_order_acquire) != g) {
          seen = g - 1;
          continue;
        }
        if (rank < j.total) {

          if (j.distinct)
            execute(j, rank);
          else
            work_chunks(j, g, rank);
        }
        continue;
      }
      for (int i = 0; i < spin; ++i) {
        if (gen_.load(std::memory_order_acquire) != seen)
          break;
        cpu_relax();
      }
      if (gen_.load(std::memory_order_acquire) != seen)
        continue;
      std::unique_lock<std::mutex> lk(m_);
      sleepers_.fetch_add(1, std::memory_order_seq_cst);
      cv_.wait(lk, [&] { return gen_.load(std::memory_order_seq_cst) != seen; });
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
  alignas(64) std::atomic<std::uint64_t> gen_{0};
  alignas(64) std::atomic<std::size_t> done_{0};
  alignas(64) std::atomic<std::size_t> sleepers_{0};
  std::atomic<bool> caller_waiting_{false};
  std::atomic<void (*)(void *, std::size_t)> call_{nullptr};
  std::atomic<void *> ctx_{nullptr};
  std::atomic<bool> distinct_{false};
  std::atomic<std::size_t> total_{0};
  std::unique_ptr<claim_slot[]> claims_;
  std::size_t nthreads_ = 0;
  std::mutex err_m_;
  std::exception_ptr err_;
};

template <typename F> void host_parallel_chunks(std::size_t n, F &&fn) {
  if (n == 0)
    return;
  std::size_t chunks = host_core_count();
  if (chunks > n)
    chunks = n;
  if (chunks > host_pool::kMaxChunks)
    chunks = host_pool::kMaxChunks;
  const std::size_t base = n / chunks, extra = n % chunks;
  host_pool::instance().run_chunks(chunks, [&](std::size_t c) {
    const std::size_t s = c * base + (c < extra ? c : extra);
    fn(s, s + base + (c < extra ? 1 : 0));
  });
}

}

#endif
#endif

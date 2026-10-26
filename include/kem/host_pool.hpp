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
    std::lock_guard<std::mutex> one_job(submit_);

    grow(slots - 1);
    auto j = std::make_shared<job>();
    j->fn = [&fn](std::size_t s) { fn(s); };
    j->slots = slots;
    j->next.store(1, std::memory_order_relaxed);
    j->remaining.store(slots - 1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(m_);
      cur_ = j;
      ++gen_;
      agen_.store(gen_, std::memory_order_release);
    }
    cv_.notify_all();
    std::exception_ptr own;
    {
      in_pool_scope scope;
      try {
        fn(std::size_t(0));
      } catch (...) {
        own = std::current_exception();
      }
    }

    const int spin = host_spin_count();
    for (int i = 0; i < spin; ++i) {
      if (j->remaining.load(std::memory_order_acquire) == 0)
        break;
      cpu_relax();
    }
    {
      std::unique_lock<std::mutex> lk(m_);
      done_cv_.wait(lk, [&] {
        return j->remaining.load(std::memory_order_acquire) == 0;
      });
      if (cur_ == j)
        cur_.reset();
    }
    if (own)
      std::rethrow_exception(own);
    if (j->err)
      std::rethrow_exception(j->err);
  }

  std::size_t size() {
    std::lock_guard<std::mutex> lk(m_);
    return nthreads_;
  }

private:
  struct job {
    std::function<void(std::size_t)> fn;
    std::size_t slots = 0;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> remaining{0};
    std::mutex err_m;
    std::exception_ptr err;
  };

  host_pool() = default;

  struct in_pool_scope {
    bool prev = tl_in_host_pool;
    in_pool_scope() { tl_in_host_pool = true; }
    ~in_pool_scope() { tl_in_host_pool = prev; }
  };

  void grow(std::size_t want) {
    std::lock_guard<std::mutex> lk(m_);
    while (nthreads_ < want) {
      std::thread([this] { loop(); }).detach();
      ++nthreads_;
    }
  }

  void loop() {
    tl_in_host_pool = true;
    std::uint64_t seen = 0;
    {

      std::lock_guard<std::mutex> lk(m_);
      seen = cur_ ? gen_ - 1 : gen_;
    }
    const int spin = host_spin_count();
    for (;;) {
      for (int i = 0; i < spin; ++i) {
        if (agen_.load(std::memory_order_acquire) != seen)
          break;
        cpu_relax();
      }
      std::shared_ptr<job> j;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        j = cur_;
      }
      if (!j)
        continue;

      const std::size_t s = j->next.fetch_add(1, std::memory_order_relaxed);
      if (s >= j->slots)
        continue;
      try {
        j->fn(s);
      } catch (...) {
        std::lock_guard<std::mutex> g(j->err_m);
        if (!j->err)
          j->err = std::current_exception();
      }
      if (j->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lk(m_);
        done_cv_.notify_all();
      }
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
  std::shared_ptr<job> cur_;
  std::uint64_t gen_ = 0;
  std::atomic<std::uint64_t> agen_{0};
  std::size_t nthreads_ = 0;
};

template <typename F> void host_parallel_chunks(std::size_t n, F &&fn) {
  if (n == 0)
    return;
  const std::size_t nt = host_core_count();
  const std::size_t chunks = n < nt ? n : nt;
  const std::size_t chunk = (n + chunks - 1) / chunks;
  const std::size_t used = (n + chunk - 1) / chunk;
  if (used == 1) {
    fn(std::size_t(0), n);
    return;
  }
  host_pool::instance().run(used, [&](std::size_t c) {
    const std::size_t b = c * chunk;
    const std::size_t e = b + chunk < n ? b + chunk : n;
    fn(b, e);
  });
}

}

#endif
#endif

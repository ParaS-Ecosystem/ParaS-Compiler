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

#ifndef __PARAS_ATOMIC_REF_HPP__
#define __PARAS_ATOMIC_REF_HPP__

#include "access.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if !PARAS_GPU_BACKEND
#include <atomic>
#endif

namespace sycl {

enum class memory_order { relaxed, acquire, release, acq_rel, seq_cst };
enum class memory_scope { work_item, sub_group, work_group, device, system };

template <memory_order> struct memory_order_traits;

template <> struct memory_order_traits<memory_order::relaxed> {
  static constexpr memory_order read_order = memory_order::relaxed;
  static constexpr memory_order write_order = memory_order::relaxed;
};

template <> struct memory_order_traits<memory_order::acq_rel> {
  static constexpr memory_order read_order = memory_order::acquire;
  static constexpr memory_order write_order = memory_order::release;
};

template <> struct memory_order_traits<memory_order::seq_cst> {
  static constexpr memory_order read_order = memory_order::seq_cst;
  static constexpr memory_order write_order = memory_order::seq_cst;
};

#if !PARAS_GPU_BACKEND
inline constexpr std::memory_order to_std(memory_order o) {
  switch (o) {
  case memory_order::relaxed:
    return std::memory_order_relaxed;
  case memory_order::acquire:
    return std::memory_order_acquire;
  case memory_order::release:
    return std::memory_order_release;
  case memory_order::acq_rel:
    return std::memory_order_acq_rel;
  case memory_order::seq_cst:
    return std::memory_order_seq_cst;
  }
  return std::memory_order_seq_cst;
}

inline constexpr std::memory_order to_std_failure(memory_order o) {
  switch (o) {
  case memory_order::release:
    return std::memory_order_relaxed;
  case memory_order::acq_rel:
    return std::memory_order_acquire;
  default:
    return to_std(o);
  }
}
#endif

#if PARAS_GPU_BACKEND

template <size_t Bytes> struct paras_cas_uint;
template <> struct paras_cas_uint<4> { using type = unsigned int; };
template <> struct paras_cas_uint<8> { using type = unsigned long long; };

template <typename To, typename From>
PARAS_KERNEL_HD To reinterpret_bits(const From &src) noexcept {
  static_assert(sizeof(To) == sizeof(From),
                "reinterpret_bits requires matching sizes");
  To dst;
  __builtin_memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename U>
PARAS_KERNEL_HD inline U paras_atomic_cas(U *p, U expected, U desired) noexcept {
#if defined(__CUDA_ARCH__)
  if (__nvvm_isspacep_local(p)) {
    const U old = *p;
    if (old == expected)
      *p = desired;
    return old;
  }
  return atomicCAS(p, expected, desired);
#elif PARAS_DEVICE_PASS
  return atomicCAS(p, expected, desired);
#else
  __atomic_compare_exchange_n(p, &expected, desired, false, __ATOMIC_SEQ_CST,
                              __ATOMIC_SEQ_CST);
  return expected;
#endif
}
#endif

PARAS_KERNEL_HD inline void atomic_fence(memory_order order, memory_scope scope) {
#if defined(__HIP_DEVICE_COMPILE__)
  if (order == memory_order::relaxed)
    return;
  if (scope == memory_scope::work_item || scope == memory_scope::sub_group ||
      scope == memory_scope::work_group)
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
  else
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
#elif defined(__CUDA_ARCH__)
  if (order == memory_order::relaxed)
    return;
  if (scope == memory_scope::work_item || scope == memory_scope::sub_group ||
      scope == memory_scope::work_group)
    __threadfence_block();
  else if (scope == memory_scope::device)
    __threadfence();
  else
    __threadfence_system();
#else
  (void)scope;
  switch (order) {
  case memory_order::relaxed:
    break;
  case memory_order::acquire:
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    break;
  case memory_order::release:
    __atomic_thread_fence(__ATOMIC_RELEASE);
    break;
  case memory_order::acq_rel:
    __atomic_thread_fence(__ATOMIC_ACQ_REL);
    break;
  default:
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
#endif
}

template <typename T, memory_order DefaultOrder = memory_order::relaxed,
          memory_scope DefaultScope = memory_scope::device,
          access::address_space AddressSpace =
              access::address_space::generic_space>
class atomic_ref {
public:
  using value_type = T;

  using difference_type =
      std::conditional_t<std::is_pointer_v<T>, std::ptrdiff_t, T>;
  static constexpr size_t required_alignment = alignof(T);

#if PARAS_GPU_BACKEND
  static constexpr bool is_always_lock_free = true;
#else
  static constexpr bool is_always_lock_free =
      std::atomic<T>::is_always_lock_free;
#endif

  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

private:
  T *ptr;

#if !PARAS_GPU_BACKEND
  std::atomic<T> *atomic_ptr() const noexcept {
    return reinterpret_cast<std::atomic<T> *>(ptr);
  }
#endif

#if PARAS_GPU_BACKEND

  template <typename V, typename BinOp>
  PARAS_KERNEL_HD T cas_loop_word(V v, BinOp op) const noexcept {
    using U = typename paras_cas_uint<sizeof(T)>::type;
    U *uptr = reinterpret_cast<U *>(ptr);
    U old_bits = *uptr;
    U assumed_bits;
    do {
      assumed_bits = old_bits;
      T assumed_val = reinterpret_bits<T>(assumed_bits);
      T new_val = op(assumed_val, v);
      U new_bits = reinterpret_bits<U>(new_val);
      old_bits = paras_atomic_cas(uptr, assumed_bits, new_bits);
    } while (assumed_bits != old_bits);
    return reinterpret_bits<T>(old_bits);
  }

  template <typename V, typename BinOp>
  PARAS_KERNEL_HD T cas_loop_masked16(V v, BinOp op) const noexcept {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = addr & ~uintptr_t(3);
    unsigned int byte_offset = static_cast<unsigned int>(addr - aligned_addr);
    unsigned int shift = byte_offset * 8;
    unsigned int mask = 0xFFFFu << shift;
    unsigned int *word_ptr = reinterpret_cast<unsigned int *>(aligned_addr);
    unsigned int old_word = *word_ptr;
    unsigned int assumed_word;

    do {
      assumed_word = old_word;
      unsigned short assumed_half =static_cast<unsigned short>((assumed_word >> shift) & 0xFFFFu);
      T assumed_val = reinterpret_bits<T>(assumed_half);
      T new_val = op(assumed_val, v);
      unsigned short new_half =reinterpret_bits<unsigned short>(new_val);
      unsigned int new_word =(assumed_word & ~mask) | (static_cast<unsigned int>(new_half) << shift);
      old_word = paras_atomic_cas(word_ptr, assumed_word, new_word);

    } while (assumed_word != old_word);

    unsigned short old_half = static_cast<unsigned short>((old_word >> shift) & 0xFFFFu);
    return reinterpret_bits<T>(old_half);
  }

  template <typename V, typename BinOp>
  PARAS_KERNEL_HD T cas_loop(V v, BinOp op) const noexcept {
    if constexpr (sizeof(T) == 2) {
      return cas_loop_masked16(v, op);
    } else {
      return cas_loop_word(v, op);
    }
  }

  template <bool Strong>

  PARAS_KERNEL_HD bool compare_exchange_word(T &expected,
                                              T desired) const noexcept {
    using U = typename paras_cas_uint<sizeof(T)>::type;
    U *uptr = reinterpret_cast<U *>(ptr);
    U expected_bits = reinterpret_bits<U>(expected);
    U desired_bits = reinterpret_bits<U>(desired);
    U old_bits = *uptr;

    do {
      if (old_bits != expected_bits) {
        expected = reinterpret_bits<T>(old_bits);
        return false;
      }
      U observed_bits = paras_atomic_cas(uptr, expected_bits, desired_bits);
      if (observed_bits == expected_bits)
        return true;
      old_bits = observed_bits;
    } while (Strong);

    expected = reinterpret_bits<T>(old_bits);
    return false;
  }

  template <bool Strong>
  PARAS_KERNEL_HD bool compare_exchange_masked16(T &expected,
                                                  T desired) const noexcept {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = addr & ~uintptr_t(3);
    unsigned int byte_offset = static_cast<unsigned int>(addr - aligned_addr);
    unsigned int shift = byte_offset * 8;
    unsigned int mask = 0xFFFFu << shift;
    unsigned int *word_ptr = reinterpret_cast<unsigned int *>(aligned_addr);
    unsigned short expected_half = reinterpret_bits<unsigned short>(expected);
    unsigned short desired_half = reinterpret_bits<unsigned short>(desired);
    unsigned int old_word = *word_ptr;

    do {
      unsigned short old_half =
          static_cast<unsigned short>((old_word >> shift) & 0xFFFFu);
      if (old_half != expected_half) {
        expected = reinterpret_bits<T>(old_half);
        return false;
      }

      unsigned int desired_word =
          (old_word & ~mask) |
          (static_cast<unsigned int>(desired_half) << shift);
      unsigned int observed_word =
          paras_atomic_cas(word_ptr, old_word, desired_word);
      if (observed_word == old_word)
        return true;
      old_word = observed_word;
    } while (Strong);

    expected = reinterpret_bits<T>(static_cast<unsigned short>(
        (old_word >> shift) & 0xFFFFu));
    return false;
  }

  template <bool Strong>
  PARAS_KERNEL_HD bool compare_exchange(T &expected, T desired) const noexcept {
    if constexpr (sizeof(T) == 2) {
      return compare_exchange_masked16<Strong>(expected, desired);
    } else {
      return compare_exchange_word<Strong>(expected, desired);
    }
  }
#endif

public:
  PARAS_KERNEL_HD
  explicit atomic_ref(T &ref) : ptr(&ref) {}
  PARAS_KERNEL_HD atomic_ref(const atomic_ref &) noexcept = default;
  atomic_ref &operator=(const atomic_ref &) = delete;

  PARAS_KERNEL_HD bool is_lock_free() const noexcept {
    return is_always_lock_free;
  }

  PARAS_KERNEL_HD static constexpr int paras_load_order(memory_order o) {
    return o == memory_order::relaxed ? __ATOMIC_RELAXED
           : (o == memory_order::acquire || o == memory_order::acq_rel)
               ? __ATOMIC_ACQUIRE
               : __ATOMIC_SEQ_CST;
  }
  PARAS_KERNEL_HD static constexpr int paras_store_order(memory_order o) {
    return o == memory_order::relaxed ? __ATOMIC_RELAXED
           : (o == memory_order::release || o == memory_order::acq_rel)
               ? __ATOMIC_RELEASE
               : __ATOMIC_SEQ_CST;
  }

  PARAS_KERNEL_HD
  void store(T v, memory_order o = default_write_order,
             memory_scope scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND

    if (paras_store_order(o) != __ATOMIC_RELAXED)
      ::sycl::atomic_fence(memory_order::seq_cst, scope);
    __atomic_store(ptr, &v, __ATOMIC_RELAXED);
    if (paras_store_order(o) == __ATOMIC_SEQ_CST)
      ::sycl::atomic_fence(memory_order::seq_cst, scope);
#else
    atomic_ptr()->store(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T load(memory_order o = default_read_order,
         memory_scope scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    T r{};
    if (paras_load_order(o) == __ATOMIC_SEQ_CST)
      ::sycl::atomic_fence(memory_order::seq_cst, scope);
    __atomic_load(ptr, &r, __ATOMIC_RELAXED);
    if (paras_load_order(o) != __ATOMIC_RELAXED)
      ::sycl::atomic_fence(memory_order::seq_cst, scope);
    return r;
#else
    return atomic_ptr()->load(to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  operator T() const noexcept { return load(); }

  PARAS_KERNEL_HD
  T operator=(T desired) const noexcept {
    store(desired);
    return desired;
  }

  PARAS_KERNEL_HD
  T exchange(T v, memory_order o = default_read_modify_write_order,
             memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    (void)o;
    return cas_loop(v, [] PARAS_KERNEL_HD(T, T b) { return b; });
#else
    return atomic_ptr()->exchange(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  bool compare_exchange_weak(
      T &expected, T desired,
      memory_order success, memory_order failure,
      memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    (void)success;
    (void)failure;
    return compare_exchange<false>(expected, desired);
#else
    return atomic_ptr()->compare_exchange_weak(expected, desired,to_std(success),to_std(failure));
#endif
  }

  PARAS_KERNEL_HD
  bool compare_exchange_weak(
      T &expected, T desired,
      memory_order order = default_read_modify_write_order,
      memory_scope scope = default_scope) const noexcept {
    memory_order failure = order;
    if (order == memory_order::acq_rel)
    failure = memory_order::acquire;
    else if (order == memory_order::release)
    failure = memory_order::relaxed;

return compare_exchange_weak(expected, desired, order, failure, scope);
  }

  PARAS_KERNEL_HD
  bool compare_exchange_strong(
      T &expected, T desired,
      memory_order success, memory_order failure,
      memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    (void)success;
    (void)failure;
    return compare_exchange<true>(expected, desired);
#else
    return atomic_ptr()->compare_exchange_strong(expected, desired,to_std(success),to_std(failure));
#endif
  }

  PARAS_KERNEL_HD
  bool compare_exchange_strong(
      T &expected, T desired,
      memory_order order = default_read_modify_write_order,
      memory_scope scope = default_scope) const noexcept {
    memory_order failure = order;
    if (order == memory_order::acq_rel)
      failure = memory_order::acquire;
    else if (order == memory_order::release)
      failure = memory_order::relaxed;
    return compare_exchange_strong(expected, desired, order, failure, scope);
  }

  PARAS_KERNEL_HD
  T fetch_add(difference_type v,
              memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return cas_loop(v, [] PARAS_KERNEL_HD(T a, difference_type b) {
      return static_cast<T>(a + b);
    });
#else
    const std::memory_order success_order = to_std(o);
    const std::memory_order failure_order = to_std_failure(o);
    T old = atomic_ptr()->load(success_order);
    T desired;
    do {
      desired = static_cast<T>(old + v);
    } while (!atomic_ptr()->compare_exchange_weak(
        old, desired, success_order, failure_order));
    return old;
#endif
  }

  PARAS_KERNEL_HD
  T fetch_sub(difference_type v,
              memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return cas_loop(v, [] PARAS_KERNEL_HD(T a, difference_type b) {
      return static_cast<T>(a - b);
    });
#else
    const std::memory_order success_order = to_std(o);
    const std::memory_order failure_order = to_std_failure(o);
    T old = atomic_ptr()->load(success_order);
    T desired;
    do {
      desired = static_cast<T>(old - v);
    } while (!atomic_ptr()->compare_exchange_weak(
        old, desired, success_order, failure_order));
    return old;
#endif
  }

  PARAS_KERNEL_HD
  T fetch_or(T v, memory_order o = default_read_modify_write_order,
             memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return cas_loop(v, [] PARAS_KERNEL_HD(T a, T b) {
      return static_cast<T>(a | b);
    });
#else
    return atomic_ptr()->fetch_or(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_xor(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return cas_loop(v, [] PARAS_KERNEL_HD(T a, T b) {
      return static_cast<T>(a ^ b);
    });
#else
    return atomic_ptr()->fetch_xor(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_and(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return cas_loop(v, [] PARAS_KERNEL_HD(T a, T b) {
      return static_cast<T>(a & b);
    });
#else
    return atomic_ptr()->fetch_and(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_min(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return cas_loop(v, [] PARAS_KERNEL_HD(T a, T b) {
      return b < a ? b : a;
    });
#else
    T old = atomic_ptr()->load(to_std(o));
    while (v < old &&
           !atomic_ptr()->compare_exchange_weak(old, v, to_std(o), to_std(o))) {
    }
    return old;
#endif
  }

  PARAS_KERNEL_HD
  T fetch_max(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return cas_loop(v, [] PARAS_KERNEL_HD(T a, T b) {
      return b > a ? b : a;
    });
#else
    T old = atomic_ptr()->load(to_std(o));
    while (v > old &&
           !atomic_ptr()->compare_exchange_weak(old, v, to_std(o), to_std(o))) {
    }
    return old;
#endif
  }

  PARAS_KERNEL_HD T operator+=(difference_type v) const noexcept {
    return static_cast<T>(fetch_add(v) + v);
  }
  PARAS_KERNEL_HD T operator-=(difference_type v) const noexcept {
    return static_cast<T>(fetch_sub(v) - v);
  }

  template <typename U = T,
            std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  PARAS_KERNEL_HD T operator++(int) const noexcept {
    return fetch_add(difference_type(1));
  }
  template <typename U = T,
            std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  PARAS_KERNEL_HD T operator++() const noexcept {
    return static_cast<T>(fetch_add(difference_type(1)) + difference_type(1));
  }
  template <typename U = T,
            std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  PARAS_KERNEL_HD T operator--(int) const noexcept {
    return fetch_sub(difference_type(1));
  }
  template <typename U = T,
            std::enable_if_t<!std::is_floating_point_v<U>, int> = 0>
  PARAS_KERNEL_HD T operator--() const noexcept {
    return static_cast<T>(fetch_sub(difference_type(1)) - difference_type(1));
  }
  PARAS_KERNEL_HD T operator|=(T v) const noexcept { return fetch_or(v) | v; }
  PARAS_KERNEL_HD T operator^=(T v) const noexcept { return fetch_xor(v) ^ v; }
  PARAS_KERNEL_HD T operator&=(T v) const noexcept { return fetch_and(v) & v; }

  PARAS_KERNEL_HD T min(T v) const noexcept { return fetch_min(v); }
  PARAS_KERNEL_HD T max(T v) const noexcept { return fetch_max(v); }
};

}

#endif

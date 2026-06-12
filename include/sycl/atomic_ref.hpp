/**<
 * Copyright 2025 The ParaS-Compiler Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_ATOMIC_REF_HPP__
#define __PARAS_ATOMIC_REF_HPP__

#include "access.hpp"
#include <cstddef>
#include <type_traits>

#if !PARAS_GPU_BACKEND
#include <atomic>
#endif

#define PARAS_KERNEL_D
#define PARAS_KERNEL_HD

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
#endif

template <typename T, memory_order DefaultOrder = memory_order::relaxed,
          memory_scope DefaultScope = memory_scope::device,
          access::address_space AddressSpace =
              access::address_space::generic_space>
class atomic_ref {
public:
  using value_type = T;
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

public:
  PARAS_KERNEL_HD
  explicit atomic_ref(T &ref) : ptr(&ref) {}

  PARAS_KERNEL_HD
  void store(T v, memory_order o = default_write_order,
             memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    *ptr = v;
#else
    atomic_ptr()->store(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T load(memory_order o = default_read_order,
         memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return *ptr;
#else
    return atomic_ptr()->load(to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  operator T() const noexcept { return load(); }

  PARAS_KERNEL_HD
  T fetch_add(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return atomicAdd(ptr, v);
#else
    return atomic_ptr()->fetch_add(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_sub(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return atomicAdd(ptr, -v);
#else
    return atomic_ptr()->fetch_sub(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_or(T v, memory_order o = default_read_modify_write_order,
             memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return atomicOr(ptr, v);
#else
    return atomic_ptr()->fetch_or(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_xor(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return atomicXor(ptr, v);
#else
    return atomic_ptr()->fetch_xor(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_and(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return atomicAnd(ptr, v);
#else
    return atomic_ptr()->fetch_and(v, to_std(o));
#endif
  }

  PARAS_KERNEL_HD
  T fetch_min(T v, memory_order o = default_read_modify_write_order,
              memory_scope = default_scope) const noexcept {
#if PARAS_GPU_BACKEND
    return atomicMin(ptr, v);
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
    return atomicMax(ptr, v);
#else
    T old = atomic_ptr()->load(to_std(o));
    while (v > old &&
           !atomic_ptr()->compare_exchange_weak(old, v, to_std(o), to_std(o))) {
    }
    return old;
#endif
  }

  PARAS_KERNEL_HD T operator+=(T v) const noexcept { return fetch_add(v) + v; }
  PARAS_KERNEL_HD T operator-=(T v) const noexcept { return fetch_sub(v) - v; }
  PARAS_KERNEL_HD T operator|=(T v) const noexcept { return fetch_or(v) | v; }
  PARAS_KERNEL_HD T operator^=(T v) const noexcept { return fetch_xor(v) ^ v; }
  PARAS_KERNEL_HD T operator&=(T v) const noexcept { return fetch_and(v) & v; }

  PARAS_KERNEL_HD T min(T v) const noexcept { return fetch_min(v); }
  PARAS_KERNEL_HD T max(T v) const noexcept { return fetch_max(v); }
};

} // namespace sycl

#endif

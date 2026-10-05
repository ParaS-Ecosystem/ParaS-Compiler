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

#include <cstdint>
#include "kem_gpu/gpu_utilities.hpp"
#include <type_traits>
#include "access.hpp"
#include <cstddef>

#if !PARAS_GPU_BACKEND
#include <atomic>
#endif

namespace sycl {

/* ================= enums ================= */

enum class memory_order { relaxed, acquire, release, acq_rel, seq_cst };
enum class memory_scope { work_item, sub_group, work_group, device, system };

/* ================= order traits ================= */

template <memory_order>
struct memory_order_traits;

template <>
struct memory_order_traits<memory_order::relaxed> {
    static constexpr memory_order read_order = memory_order::relaxed;
    static constexpr memory_order write_order = memory_order::relaxed;
};

template <>
struct memory_order_traits<memory_order::acq_rel> {
    static constexpr memory_order read_order = memory_order::acquire;
    static constexpr memory_order write_order = memory_order::release;
};

template <>
struct memory_order_traits<memory_order::seq_cst> {
    static constexpr memory_order read_order = memory_order::seq_cst;
    static constexpr memory_order write_order = memory_order::seq_cst;
};

template <>
struct memory_order_traits<memory_order::acquire> {
    static constexpr memory_order read_order = memory_order::acquire;
    static constexpr memory_order write_order = memory_order::relaxed;
};

template <>
struct memory_order_traits<memory_order::release> {
    static constexpr memory_order read_order = memory_order::relaxed;
    static constexpr memory_order write_order = memory_order::release;
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

PARAS_KERNEL_HD
inline void atomic_fence(memory_order order, memory_scope scope) noexcept {
#if PARAS_GPU_BACKEND
    (void)order;
    switch (scope) {
    case memory_scope::work_item:
    case memory_scope::sub_group:
    case memory_scope::work_group:
        __threadfence_block();
        break;
    case memory_scope::device:
        __threadfence();
        break;
    case memory_scope::system:
        __threadfence_system();
        break;
    }
#else
    (void)scope;
    std::atomic_thread_fence(to_std(order));
#endif
}

/* ================= atomic_ref ================= */
namespace paras_atomic_detail {
PARAS_KERNEL_HD
inline constexpr int to_builtin(memory_order o) {
    return o == memory_order::relaxed   ? __ATOMIC_RELAXED
           : o == memory_order::acquire ? __ATOMIC_ACQUIRE
           : o == memory_order::release ? __ATOMIC_RELEASE
           : o == memory_order::acq_rel ? __ATOMIC_ACQ_REL
                                        : __ATOMIC_SEQ_CST;
}
PARAS_KERNEL_HD
inline constexpr int to_builtin_failure(memory_order o) {
    return o == memory_order::release   ? __ATOMIC_RELAXED
           : o == memory_order::acq_rel ? __ATOMIC_ACQUIRE
                                        : to_builtin(o);
}
} // namespace paras_atomic_detail

template <typename T, memory_order DefaultOrder = memory_order::relaxed,
          memory_scope DefaultScope = memory_scope::device,
          access::address_space AddressSpace = access::address_space::generic_space>
class atomic_ref {
public:
    using value_type = T;
    using difference_type = std::conditional_t<std::is_pointer_v<T>, std::ptrdiff_t, T>;
    using operand_type = difference_type;

    static constexpr size_t required_alignment = alignof(T);
    static constexpr bool is_always_lock_free = __atomic_always_lock_free(sizeof(T), 0);

    PARAS_KERNEL_HD
    bool is_lock_free() const noexcept { return is_always_lock_free; }

    static constexpr memory_order default_read_order =
        memory_order_traits<DefaultOrder>::read_order;
    static constexpr memory_order default_write_order =
        memory_order_traits<DefaultOrder>::write_order;
    static constexpr memory_order default_read_modify_write_order = DefaultOrder;
    static constexpr memory_scope default_scope = DefaultScope;

private:
    T* ptr;

    template <typename Op>
    PARAS_KERNEL_HD T cas_loop(Op op, memory_order o) const noexcept {
        T old = load(memory_order::relaxed);
        T desired = op(old);
        while (!compare_exchange_weak(old, desired, o, memory_order::relaxed)) {
            desired = op(old);
        }
        return old;
    }

public:
    PARAS_KERNEL_HD
    explicit atomic_ref(T& ref) : ptr(&ref) {}

    PARAS_KERNEL_HD
    atomic_ref(const atomic_ref&) noexcept = default;

    /* ================= basic ops ================= */
    PARAS_KERNEL_HD
    void store(T v, memory_order o = default_write_order,
               memory_scope = default_scope) const noexcept {
        __atomic_store(ptr, &v, paras_atomic_detail::to_builtin(o));
    }

    PARAS_KERNEL_HD
    T load(memory_order o = default_read_order, memory_scope = default_scope) const noexcept {
        T r;
        __atomic_load(ptr, &r, paras_atomic_detail::to_builtin(o));
        return r;
    }

    PARAS_KERNEL_HD
    T exchange(T v, memory_order o = default_read_modify_write_order,
               memory_scope = default_scope) const noexcept {
        T r;
        __atomic_exchange(ptr, &v, &r, paras_atomic_detail::to_builtin(o));
        return r;
    }

    PARAS_KERNEL_HD
    operator T() const noexcept { return load(); }

    PARAS_KERNEL_HD
    T operator=(T v) const noexcept {
        store(v);
        return v;
    }

    /* ================= compare_exchange ================= */
    PARAS_KERNEL_HD
    bool compare_exchange_weak(T& expected, T desired, memory_order success, memory_order failure,
                               memory_scope = default_scope) const noexcept {
        return __atomic_compare_exchange(ptr, &expected, &desired, true,
                                         paras_atomic_detail::to_builtin(success),
                                         paras_atomic_detail::to_builtin_failure(failure));
    }

    PARAS_KERNEL_HD
    bool compare_exchange_weak(T& expected, T desired,
                               memory_order order = default_read_modify_write_order,
                               memory_scope scope = default_scope) const noexcept {
        return compare_exchange_weak(expected, desired, order, order, scope);
    }

    PARAS_KERNEL_HD
    bool compare_exchange_strong(T& expected, T desired, memory_order success, memory_order failure,
                                 memory_scope = default_scope) const noexcept {
        return __atomic_compare_exchange(ptr, &expected, &desired, false,
                                         paras_atomic_detail::to_builtin(success),
                                         paras_atomic_detail::to_builtin_failure(failure));
    }

    PARAS_KERNEL_HD
    bool compare_exchange_strong(T& expected, T desired,
                                 memory_order order = default_read_modify_write_order,
                                 memory_scope scope = default_scope) const noexcept {
        return compare_exchange_strong(expected, desired, order, order, scope);
    }

    /* ================= arithmetic ================= */
    PARAS_KERNEL_HD
    T fetch_add(operand_type v, memory_order o = default_read_modify_write_order,
                memory_scope = default_scope) const noexcept {
        if constexpr (std::is_pointer_v<T>) {
            return cas_loop([v](T old) { return old + v; }, o);
        } else {
            return __atomic_fetch_add(ptr, v, paras_atomic_detail::to_builtin(o));
        }
    }

    PARAS_KERNEL_HD
    T fetch_sub(operand_type v, memory_order o = default_read_modify_write_order,
                memory_scope = default_scope) const noexcept {
        if constexpr (std::is_pointer_v<T>) {
            return cas_loop([v](T old) { return old - v; }, o);
        } else {
            return __atomic_fetch_sub(ptr, v, paras_atomic_detail::to_builtin(o));
        }
    }

    /* ================= bitwise (integral only) ================= */
    PARAS_KERNEL_HD
    T fetch_or(T v, memory_order o = default_read_modify_write_order,
               memory_scope = default_scope) const noexcept {
        return __atomic_fetch_or(ptr, v, paras_atomic_detail::to_builtin(o));
    }

    PARAS_KERNEL_HD
    T fetch_xor(T v, memory_order o = default_read_modify_write_order,
                memory_scope = default_scope) const noexcept {
        return __atomic_fetch_xor(ptr, v, paras_atomic_detail::to_builtin(o));
    }

    PARAS_KERNEL_HD
    T fetch_and(T v, memory_order o = default_read_modify_write_order,
                memory_scope = default_scope) const noexcept {
        return __atomic_fetch_and(ptr, v, paras_atomic_detail::to_builtin(o));
    }

    /* ================= min / max ================= */
    PARAS_KERNEL_HD
    T fetch_min(T v, memory_order o = default_read_modify_write_order,
                memory_scope = default_scope) const noexcept {
        if constexpr (std::is_integral_v<T>) {
            return __atomic_fetch_min(ptr, v, paras_atomic_detail::to_builtin(o));
        } else {
            return cas_loop([v](T old) { return v < old ? v : old; }, o);
        }
    }

    PARAS_KERNEL_HD
    T fetch_max(T v, memory_order o = default_read_modify_write_order,
                memory_scope = default_scope) const noexcept {
        if constexpr (std::is_integral_v<T>) {
            return __atomic_fetch_max(ptr, v, paras_atomic_detail::to_builtin(o));
        } else {
            return cas_loop([v](T old) { return v > old ? v : old; }, o);
        }
    }

    /* ================= operators ================= */
    PARAS_KERNEL_HD T operator+=(operand_type v) const noexcept { return fetch_add(v) + v; }
    PARAS_KERNEL_HD T operator-=(operand_type v) const noexcept { return fetch_sub(v) - v; }
    PARAS_KERNEL_HD T operator|=(T v) const noexcept { return fetch_or(v) | v; }
    PARAS_KERNEL_HD T operator^=(T v) const noexcept { return fetch_xor(v) ^ v; }
    PARAS_KERNEL_HD T operator&=(T v) const noexcept { return fetch_and(v) & v; }

    PARAS_KERNEL_HD T operator++() const noexcept { return fetch_add(1) + 1; }
    PARAS_KERNEL_HD T operator++(int) const noexcept { return fetch_add(1); }
    PARAS_KERNEL_HD T operator--() const noexcept { return fetch_sub(1) - 1; }
    PARAS_KERNEL_HD T operator--(int) const noexcept { return fetch_sub(1); }

    PARAS_KERNEL_HD T min(T v) const noexcept { return fetch_min(v); }
    PARAS_KERNEL_HD T max(T v) const noexcept { return fetch_max(v); }
};

} // namespace sycl

#endif

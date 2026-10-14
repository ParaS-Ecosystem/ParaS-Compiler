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

#ifndef __PARAS_ATOMIC_HPP__
#define __PARAS_ATOMIC_HPP__

#include <type_traits>
#include "access.hpp"
#include "atomic_ref.hpp"
#include "multi_ptr.hpp"
#include "kem_gpu/gpu_utilities.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace sycl {

template <typename T, access::address_space AddressSpace = access::address_space::global_space>
class [[deprecated("sycl::atomic is deprecated in SYCL 2020, use sycl::atomic_ref")]] atomic {
    template <typename U>
    static constexpr bool not_float = !std::is_same_v<U, float>;

public:
    template <typename PointerT, access::decorated IsDecorated,
              typename = std::enable_if_t<sizeof(PointerT) == sizeof(T)>>
    PARAS_KERNEL_HD atomic(multi_ptr<PointerT, AddressSpace, IsDecorated> ptr)
        : ptr_(reinterpret_cast<T*>(ptr.get_raw())) {}

    PARAS_KERNEL_HD void store(T operand, memory_order = memory_order::relaxed) {
        __atomic_store(ptr_, &operand, __ATOMIC_RELAXED);
    }
    PARAS_KERNEL_HD T load(memory_order = memory_order::relaxed) const {
        T v;
        __atomic_load(ptr_, &v, __ATOMIC_RELAXED);
        return v;
    }
    PARAS_KERNEL_HD T exchange(T operand, memory_order = memory_order::relaxed) {
        T old;
        __atomic_exchange(ptr_, &operand, &old, __ATOMIC_RELAXED);
        return old;
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD bool compare_exchange_strong(T& expected, T desired,
                                                 memory_order = memory_order::relaxed,
                                                 memory_order = memory_order::relaxed) {
        return __atomic_compare_exchange(ptr_, &expected, &desired, false, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED);
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD T fetch_add(T operand, memory_order = memory_order::relaxed) {
        return __atomic_fetch_add(ptr_, operand, __ATOMIC_RELAXED);
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD T fetch_sub(T operand, memory_order = memory_order::relaxed) {
        return __atomic_fetch_sub(ptr_, operand, __ATOMIC_RELAXED);
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD T fetch_and(T operand, memory_order = memory_order::relaxed) {
        return __atomic_fetch_and(ptr_, operand, __ATOMIC_RELAXED);
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD T fetch_or(T operand, memory_order = memory_order::relaxed) {
        return __atomic_fetch_or(ptr_, operand, __ATOMIC_RELAXED);
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD T fetch_xor(T operand, memory_order = memory_order::relaxed) {
        return __atomic_fetch_xor(ptr_, operand, __ATOMIC_RELAXED);
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD T fetch_min(T operand, memory_order = memory_order::relaxed) {
        T old = load();
        while (operand < old &&
               !__atomic_compare_exchange(ptr_, &old, &operand, true, __ATOMIC_RELAXED,
                                          __ATOMIC_RELAXED)) {
        }
        return old;
    }
    template <typename U = T, typename = std::enable_if_t<not_float<U>>>
    PARAS_KERNEL_HD T fetch_max(T operand, memory_order = memory_order::relaxed) {
        T old = load();
        while (operand > old &&
               !__atomic_compare_exchange(ptr_, &old, &operand, true, __ATOMIC_RELAXED,
                                          __ATOMIC_RELAXED)) {
        }
        return old;
    }

private:
    T* ptr_;
};

template <typename T, access::address_space AddressSpace>
PARAS_KERNEL_HD void atomic_store(atomic<T, AddressSpace> object, T operand,
                                  memory_order memoryOrder = memory_order::relaxed) {
    object.store(operand, memoryOrder);
}
template <typename T, access::address_space AddressSpace>
PARAS_KERNEL_HD T atomic_load(atomic<T, AddressSpace> object,
                              memory_order memoryOrder = memory_order::relaxed) {
    return object.load(memoryOrder);
}
template <typename T, access::address_space AddressSpace>
PARAS_KERNEL_HD T atomic_exchange(atomic<T, AddressSpace> object, T operand,
                                  memory_order memoryOrder = memory_order::relaxed) {
    return object.exchange(operand, memoryOrder);
}
template <typename T, access::address_space AddressSpace>
PARAS_KERNEL_HD bool atomic_compare_exchange_strong(
    atomic<T, AddressSpace> object, T& expected, T desired,
    memory_order successMemoryOrder = memory_order::relaxed,
    memory_order failMemoryOrder = memory_order::relaxed) {
    return object.compare_exchange_strong(expected, desired, successMemoryOrder, failMemoryOrder);
}
#define PARAS_ATOMIC_FETCH_FN(NAME)                                                                \
    template <typename T, access::address_space AddressSpace>                                      \
    PARAS_KERNEL_HD T atomic_##NAME(atomic<T, AddressSpace> object, T operand,                     \
                                    memory_order memoryOrder = memory_order::relaxed) {            \
        return object.NAME(operand, memoryOrder);                                                  \
    }
PARAS_ATOMIC_FETCH_FN(fetch_add)
PARAS_ATOMIC_FETCH_FN(fetch_sub)
PARAS_ATOMIC_FETCH_FN(fetch_and)
PARAS_ATOMIC_FETCH_FN(fetch_or)
PARAS_ATOMIC_FETCH_FN(fetch_xor)
PARAS_ATOMIC_FETCH_FN(fetch_min)
PARAS_ATOMIC_FETCH_FN(fetch_max)
#undef PARAS_ATOMIC_FETCH_FN

}

#pragma clang diagnostic pop

#endif

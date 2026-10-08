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

#include "access.hpp"
#include "atomic_ref.hpp"
#include "multi_ptr.hpp"
#include "kem_gpu/gpu_utilities.hpp"
#include <type_traits>

namespace sycl {

template <typename T, access::address_space AddressSpace =
                          access::address_space::global_space>
class atomic {
  T *ptr_;
  using ref_t = atomic_ref<T, memory_order::relaxed, memory_scope::device,
                           access::address_space::generic_space>;
  PARAS_KERNEL_HD ref_t ref() const { return ref_t(*ptr_); }

public:
  using value_type = T;

  template <typename PointerT, access::decorated IsDecorated>
  PARAS_KERNEL_HD atomic(multi_ptr<PointerT, AddressSpace, IsDecorated> p)
      : ptr_(reinterpret_cast<T *>(p.get())) {
    static_assert(sizeof(PointerT) == sizeof(T),
                  "atomic: pointer type must match the atomic type");
  }

  PARAS_KERNEL_HD atomic(T &v) : ptr_(&v) {}

  PARAS_KERNEL_HD void store(T operand,
                             memory_order o = memory_order::relaxed) {
    ref().store(operand, o);
  }
  PARAS_KERNEL_HD T load(memory_order o = memory_order::relaxed) const {
    return ref().load(o);
  }
  PARAS_KERNEL_HD T exchange(T operand,
                             memory_order o = memory_order::relaxed) {
    return ref().exchange(operand, o);
  }
  PARAS_KERNEL_HD bool
  compare_exchange_strong(T &expected, T desired,
                          memory_order success = memory_order::relaxed,
                          memory_order fail = memory_order::relaxed) {
    return ref().compare_exchange_strong(expected, desired, success, fail);
  }
#define PARAS_LEGACY_ATOMIC_OP(name)                                          \
  PARAS_KERNEL_HD T name(T operand, memory_order o = memory_order::relaxed) { \
    return ref().name(operand, o);                                            \
  }
  PARAS_LEGACY_ATOMIC_OP(fetch_add)
  PARAS_LEGACY_ATOMIC_OP(fetch_sub)
  PARAS_LEGACY_ATOMIC_OP(fetch_and)
  PARAS_LEGACY_ATOMIC_OP(fetch_or)
  PARAS_LEGACY_ATOMIC_OP(fetch_xor)
  PARAS_LEGACY_ATOMIC_OP(fetch_min)
  PARAS_LEGACY_ATOMIC_OP(fetch_max)
#undef PARAS_LEGACY_ATOMIC_OP
};

template <typename T, access::address_space AS>
PARAS_KERNEL_HD T atomic_load(atomic<T, AS> object,
                              memory_order o = memory_order::relaxed) {
  return object.load(o);
}
template <typename T, access::address_space AS>
PARAS_KERNEL_HD void atomic_store(atomic<T, AS> object,
                                  typename std::common_type<T>::type operand,
                                  memory_order o = memory_order::relaxed) {
  object.store(operand, o);
}
template <typename T, access::address_space AS>
PARAS_KERNEL_HD T atomic_exchange(atomic<T, AS> object,
                                  typename std::common_type<T>::type operand,
                                  memory_order o = memory_order::relaxed) {
  return object.exchange(operand, o);
}
template <typename T, access::address_space AS>
PARAS_KERNEL_HD bool atomic_compare_exchange_strong(
    atomic<T, AS> object, T &expected,
    typename std::common_type<T>::type desired,
    memory_order success = memory_order::relaxed,
    memory_order fail = memory_order::relaxed) {
  return object.compare_exchange_strong(expected, desired, success, fail);
}
#define PARAS_LEGACY_ATOMIC_FREE(name)                                        \
  template <typename T, access::address_space AS>                             \
  PARAS_KERNEL_HD T atomic_##name(atomic<T, AS> object,                       \
                                  typename std::common_type<T>::type operand, \
                                  memory_order o = memory_order::relaxed) {   \
    return object.name(operand, o);                                           \
  }
PARAS_LEGACY_ATOMIC_FREE(fetch_add)
PARAS_LEGACY_ATOMIC_FREE(fetch_sub)
PARAS_LEGACY_ATOMIC_FREE(fetch_and)
PARAS_LEGACY_ATOMIC_FREE(fetch_or)
PARAS_LEGACY_ATOMIC_FREE(fetch_xor)
PARAS_LEGACY_ATOMIC_FREE(fetch_min)
PARAS_LEGACY_ATOMIC_FREE(fetch_max)
#undef PARAS_LEGACY_ATOMIC_FREE

}

#endif

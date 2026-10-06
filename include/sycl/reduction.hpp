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

#ifndef __PARAS_REDUCTION_HPP__
#define __PARAS_REDUCTION_HPP__

#include <cstddef>
#include <type_traits>
#include "kem_gpu/gpu_utilities.hpp"
#include "known_identity.hpp"
#include "property_list.hpp"

namespace sycl {

template <typename T, typename BinaryOperation>
class reduction_impl {
public:
    T* var_;
    BinaryOperation op_;
    T identity_;
    bool identity_known_;
    bool initialize_to_identity_;

    reduction_impl(T* varPtr, BinaryOperation op, T identity, bool identity_known,
                   bool initialize_to_identity)
        : var_(varPtr), op_(op), identity_(identity), identity_known_(identity_known),
          initialize_to_identity_(initialize_to_identity) {}
};

namespace paras_reduction_detail {
template <typename BinaryOperation, typename T>
constexpr T identity_or_default() {
    if constexpr (has_known_identity_v<BinaryOperation, T>) {
        return known_identity_v<BinaryOperation, T>;
    } else {
        return T{};
    }
}
} // namespace paras_reduction_detail

template <typename T, typename BinaryOperation>
reduction_impl<T, BinaryOperation> reduction(T* varPtr, BinaryOperation combiner,
                                             const property_list& props = {}) {
    return reduction_impl<T, BinaryOperation>(
        varPtr, combiner, paras_reduction_detail::identity_or_default<BinaryOperation, T>(),
        has_known_identity_v<BinaryOperation, T>, props.has_initialize_to_identity());
}

template <typename T, typename BinaryOperation>
reduction_impl<T, BinaryOperation> reduction(T* varPtr, const T& identity, BinaryOperation combiner,
                                             const property_list& props = {}) {
    return reduction_impl<T, BinaryOperation>(varPtr, combiner, identity, true,
                                              props.has_initialize_to_identity());
}

template <typename T, typename BinaryOperation>
class reducer {
    T value_;
    T identity_;
    BinaryOperation op_;
    bool has_value_;

public:
    using value_type = T;
    using binary_operation = BinaryOperation;
    static constexpr int dimensions = 0;

    PARAS_KERNEL_HD
    reducer(const T& identity, bool identity_known, BinaryOperation op)
        : value_(identity), identity_(identity), op_(op), has_value_(identity_known) {}

    reducer(const reducer&) = delete;
    reducer& operator=(const reducer&) = delete;

    PARAS_KERNEL_HD
    reducer& combine(const T& partial) {
        value_ = has_value_ ? op_(value_, partial) : partial;
        has_value_ = true;
        return *this;
    }

    PARAS_KERNEL_HD T identity() const { return identity_; }

    PARAS_KERNEL_HD reducer& operator+=(const T& partial) { return combine(partial); }
    PARAS_KERNEL_HD reducer& operator*=(const T& partial) { return combine(partial); }
    PARAS_KERNEL_HD reducer& operator|=(const T& partial) { return combine(partial); }
    PARAS_KERNEL_HD reducer& operator&=(const T& partial) { return combine(partial); }
    PARAS_KERNEL_HD reducer& operator^=(const T& partial) { return combine(partial); }
    PARAS_KERNEL_HD reducer& operator++() { return combine(T{1}); }
    PARAS_KERNEL_HD void operator++(int) { combine(T{1}); }

    PARAS_KERNEL_HD bool paras_has_value() const { return has_value_; }
    PARAS_KERNEL_HD const T& paras_value() const { return value_; }
};

template <typename T, typename BinaryOperation>
PARAS_KERNEL_HD inline void paras_reduction_merge(T* var, const T& v, BinaryOperation op) {
    static_assert(std::is_trivially_copyable_v<T> &&
                      (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8),
                  "ParaS reductions need a trivially copyable type of 1, 2, 4 or 8 bytes");
    T expected;
    __atomic_load(var, &expected, __ATOMIC_RELAXED);
    T desired = op(expected, v);
    while (!__atomic_compare_exchange(var, &expected, &desired, true, __ATOMIC_ACQ_REL,
                                      __ATOMIC_RELAXED)) {
        desired = op(expected, v);
    }
}

} // namespace sycl

#endif

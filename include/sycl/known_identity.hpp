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

#ifndef __PARAS_KNOWN_IDENTITY_HPP__
#define __PARAS_KNOWN_IDENTITY_HPP__

#include <limits>
#include <type_traits>
#include "math.hpp"

namespace sycl {

namespace paras_identity_detail {

template <template <typename> class Op, typename BinaryOperation, typename T>
inline constexpr bool is_op_v =
    std::is_same_v<BinaryOperation, Op<T>> || std::is_same_v<BinaryOperation, Op<void>>;

template <typename BinaryOperation, typename T, typename = void>
struct identity {
    static constexpr bool known = false;
};

template <typename BinaryOperation, typename T>
struct identity<
    BinaryOperation, T,
    std::enable_if_t<is_op_v<plus, BinaryOperation, T> || is_op_v<bit_or, BinaryOperation, T> ||
                     is_op_v<bit_xor, BinaryOperation, T>>> {
    static constexpr bool known = std::is_arithmetic_v<T>;
    static constexpr T value = T{};
};

template <typename BinaryOperation, typename T>
struct identity<BinaryOperation, T, std::enable_if_t<is_op_v<multiplies, BinaryOperation, T>>> {
    static constexpr bool known = std::is_arithmetic_v<T>;
    static constexpr T value = T{1};
};

template <typename BinaryOperation, typename T>
struct identity<BinaryOperation, T, std::enable_if_t<is_op_v<bit_and, BinaryOperation, T>>> {
    static constexpr bool known = std::is_integral_v<T>;
    static constexpr T value = static_cast<T>(~T{});
};

template <typename BinaryOperation, typename T>
struct identity<BinaryOperation, T, std::enable_if_t<is_op_v<logical_and, BinaryOperation, T>>> {
    static constexpr bool known = std::is_arithmetic_v<T>;
    static constexpr T value = T{true};
};

template <typename BinaryOperation, typename T>
struct identity<BinaryOperation, T, std::enable_if_t<is_op_v<logical_or, BinaryOperation, T>>> {
    static constexpr bool known = std::is_arithmetic_v<T>;
    static constexpr T value = T{false};
};

template <typename BinaryOperation, typename T>
struct identity<BinaryOperation, T, std::enable_if_t<is_op_v<minimum, BinaryOperation, T>>> {
    static constexpr bool known = std::is_arithmetic_v<T>;
    static constexpr T value = std::numeric_limits<T>::has_infinity
                                   ? std::numeric_limits<T>::infinity()
                                   : (std::numeric_limits<T>::max)();
};

template <typename BinaryOperation, typename T>
struct identity<BinaryOperation, T, std::enable_if_t<is_op_v<maximum, BinaryOperation, T>>> {
    static constexpr bool known = std::is_arithmetic_v<T>;
    static constexpr T value = std::numeric_limits<T>::has_infinity
                                   ? -std::numeric_limits<T>::infinity()
                                   : std::numeric_limits<T>::lowest();
};

} // namespace paras_identity_detail

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity
    : std::bool_constant<paras_identity_detail::identity<std::decay_t<BinaryOperation>,
                                                         std::decay_t<AccumulatorT>>::known> {};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;

template <typename BinaryOperation, typename AccumulatorT>
struct known_identity {
    static_assert(has_known_identity_v<BinaryOperation, AccumulatorT>,
                  "known_identity: no known identity for this operation/type");
    static constexpr AccumulatorT value =
        paras_identity_detail::identity<std::decay_t<BinaryOperation>,
                                        std::decay_t<AccumulatorT>>::value;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace sycl

#endif

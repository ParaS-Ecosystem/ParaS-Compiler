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

#include "functional.hpp"
#include "half.hpp"

namespace sycl {

namespace detail {
template <typename Op, template <typename> class F, typename T>
inline constexpr bool is_functor_v =
    std::is_same_v<Op, F<T>> || std::is_same_v<Op, F<void>>;
template <typename T>
inline constexpr bool is_arith_or_half_v =
    std::is_arithmetic_v<T> || std::is_same_v<std::remove_cv_t<T>, half>;
template <typename T>
inline constexpr bool is_int_v =
    std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>;
template <typename T> constexpr T paras_min_identity() {
  if constexpr (std::numeric_limits<T>::has_infinity)
    return std::numeric_limits<T>::infinity();
  else
    return (std::numeric_limits<T>::max)();
}
template <typename T> constexpr T paras_max_identity() {
  if constexpr (std::numeric_limits<T>::has_infinity)
    return -std::numeric_limits<T>::infinity();
  else
    return std::numeric_limits<T>::lowest();
}
}

template <typename BinaryOperation, typename AccumulatorT, typename = void>
struct known_identity {};

#define PARAS_KNOWN_IDENTITY(FUNCTOR, COND, VALUE)                           \
  template <typename Op, typename T>                                          \
  struct known_identity<                                                      \
      Op, T,                                                                  \
      std::enable_if_t<detail::is_functor_v<Op, FUNCTOR, std::remove_cv_t<T>> && \
                       (COND)>> {                                             \
    static constexpr std::remove_cv_t<T> value = VALUE;                       \
  };
PARAS_KNOWN_IDENTITY(plus, detail::is_arith_or_half_v<T>, std::remove_cv_t<T>(0))
PARAS_KNOWN_IDENTITY(multiplies, detail::is_arith_or_half_v<T>, std::remove_cv_t<T>(1))
PARAS_KNOWN_IDENTITY(bit_and, detail::is_int_v<T>, static_cast<std::remove_cv_t<T>>(~std::remove_cv_t<T>(0)))
PARAS_KNOWN_IDENTITY(bit_or, detail::is_int_v<T>, std::remove_cv_t<T>(0))
PARAS_KNOWN_IDENTITY(bit_xor, detail::is_int_v<T>, std::remove_cv_t<T>(0))
PARAS_KNOWN_IDENTITY(logical_and, (std::is_same_v<std::remove_cv_t<T>, bool>), true)
PARAS_KNOWN_IDENTITY(logical_or, (std::is_same_v<std::remove_cv_t<T>, bool>), false)
PARAS_KNOWN_IDENTITY(minimum, detail::is_arith_or_half_v<T>, detail::paras_min_identity<std::remove_cv_t<T>>())
PARAS_KNOWN_IDENTITY(maximum, detail::is_arith_or_half_v<T>, detail::paras_max_identity<std::remove_cv_t<T>>())
#undef PARAS_KNOWN_IDENTITY

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;

template <typename BinaryOperation, typename AccumulatorT, typename = void>
struct has_known_identity : std::false_type {};
template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity<
    BinaryOperation, AccumulatorT,
    std::void_t<decltype(known_identity<BinaryOperation, AccumulatorT>::value)>>
    : std::true_type {};
template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;

}

#endif

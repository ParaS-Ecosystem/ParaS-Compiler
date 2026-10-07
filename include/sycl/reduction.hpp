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
#include <utility>
#if __cplusplus >= 202002L
#include <span>
#endif

#include "known_identity.hpp"
#include "property_list.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

class handler;

#if __cplusplus >= 202002L

inline constexpr std::size_t dynamic_extent = std::dynamic_extent;
template <typename T, std::size_t Extent = dynamic_extent>
using span = std::span<T, Extent>;
#endif

namespace detail {
template <typename Op, template <typename> class F>
inline constexpr bool op_is_v = false;
template <template <typename> class F, typename U>
inline constexpr bool op_is_v<F<U>, F> = true;
}

template <typename T, typename BinaryOperation, int Dimensions = 0>
class reducer;

template <typename T, typename BinaryOperation>
class reducer<T, BinaryOperation, 0> {
  T *val_;
  bool *touched_;
  BinaryOperation op_;
  T identity_;

public:
  using value_type = T;
  using binary_operation = BinaryOperation;
  static constexpr int dimensions = 0;

  PARAS_KERNEL_HD reducer(T *val, bool *touched, BinaryOperation op, T identity)
      : val_(val), touched_(touched), op_(op), identity_(identity) {}

  PARAS_KERNEL_HD reducer &combine(const T &partial) {
    if (*touched_) {
      *val_ = op_(*val_, partial);
    } else {
      *val_ = partial;
      *touched_ = true;
    }
    return *this;
  }
  PARAS_KERNEL_HD const reducer &combine(const T &partial) const {
    return const_cast<reducer *>(this)->combine(partial);
  }

  PARAS_KERNEL_HD T identity() const { return identity_; }

  template <typename O = BinaryOperation,
            std::enable_if_t<detail::op_is_v<O, plus>, int> = 0>
  PARAS_KERNEL_HD reducer &operator+=(const T &v) { return combine(v); }
  template <typename O = BinaryOperation,
            std::enable_if_t<detail::op_is_v<O, multiplies>, int> = 0>
  PARAS_KERNEL_HD reducer &operator*=(const T &v) { return combine(v); }
  template <typename O = BinaryOperation,
            std::enable_if_t<detail::op_is_v<O, bit_and>, int> = 0>
  PARAS_KERNEL_HD reducer &operator&=(const T &v) { return combine(v); }
  template <typename O = BinaryOperation,
            std::enable_if_t<detail::op_is_v<O, bit_or>, int> = 0>
  PARAS_KERNEL_HD reducer &operator|=(const T &v) { return combine(v); }
  template <typename O = BinaryOperation,
            std::enable_if_t<detail::op_is_v<O, bit_xor>, int> = 0>
  PARAS_KERNEL_HD reducer &operator^=(const T &v) { return combine(v); }
  template <typename O = BinaryOperation,
            std::enable_if_t<detail::op_is_v<O, plus> && std::is_integral_v<T> &&
                                 !std::is_same_v<T, bool>,
                             int> = 0>
  PARAS_KERNEL_HD reducer &operator++() { return combine(T(1)); }
  template <typename O = BinaryOperation,
            std::enable_if_t<detail::op_is_v<O, plus> && std::is_integral_v<T> &&
                                 !std::is_same_v<T, bool>,
                             int> = 0>
  PARAS_KERNEL_HD void operator++(int) { combine(T(1)); }
};

template <typename T, typename BinaryOperation>
class reducer<T, BinaryOperation, 1> {
  T *row_;
  bool *touched_;
  BinaryOperation op_;
  T identity_;

public:
  using value_type = T;
  using binary_operation = BinaryOperation;
  static constexpr int dimensions = 1;

  PARAS_KERNEL_HD reducer(T *row, bool *touched, BinaryOperation op, T identity)
      : row_(row), touched_(touched), op_(op), identity_(identity) {}

  PARAS_KERNEL_HD reducer<T, BinaryOperation, 0> operator[](std::size_t i) const {
    return reducer<T, BinaryOperation, 0>(row_ + i, touched_ + i, op_, identity_);
  }
  PARAS_KERNEL_HD T identity() const { return identity_; }
};

namespace detail {

template <typename T, typename BinaryOperation, bool IsSpan>
struct reduction_desc {
  using value_type = T;
  using binary_operation = BinaryOperation;
  T *target = nullptr;
  std::size_t count = 1;
  BinaryOperation op{};
  T identity{};
  bool has_identity = false;
  bool init_to_identity = false;

  T *slots = nullptr;
  bool *touched = nullptr;

  PARAS_KERNEL_HD auto make_reducer(std::size_t work_item) const {
    if constexpr (IsSpan)
      return reducer<T, BinaryOperation, 1>(slots + work_item * count,
                                            touched + work_item * count, op, identity);
    else
      return reducer<T, BinaryOperation, 0>(slots + work_item, touched + work_item,
                                            op, identity);
  }
};

template <typename T> struct is_reduction : std::false_type {};
template <typename T, typename Op, bool S>
struct is_reduction<reduction_desc<T, Op, S>> : std::true_type {};
template <typename T>
inline constexpr bool is_reduction_v = is_reduction<std::decay_t<T>>::value;

template <typename T, typename Op, bool S>
reduction_desc<T, Op, S> make_reduction(T *target, std::size_t count, Op op,
                                        const T *identity, const property_list &props) {
  reduction_desc<T, Op, S> d;
  d.target = target;
  d.count = count;
  d.op = op;
  if (identity != nullptr) {
    d.identity = *identity;
    d.has_identity = true;
  } else if constexpr (has_known_identity_v<Op, T>) {
    d.identity = known_identity_v<Op, T>;
    d.has_identity = true;
  }
  d.init_to_identity =
      props.template has_property<property::reduction::initialize_to_identity>();
  return d;
}
}

template <typename T, typename BinaryOperation>
auto reduction(T *variable, BinaryOperation combiner, const property_list &props = {}) {
  return detail::make_reduction<T, BinaryOperation, false>(variable, 1, combiner,
                                                           nullptr, props);
}
template <typename T, typename BinaryOperation>
auto reduction(T *variable, const T &identity, BinaryOperation combiner,
               const property_list &props = {}) {
  return detail::make_reduction<T, BinaryOperation, false>(variable, 1, combiner,
                                                           &identity, props);
}

template <typename BufferT, typename BinaryOperation,
          typename = decltype(std::declval<BufferT &>().get_range()),
          typename T = typename BufferT::value_type>
auto reduction(BufferT &vars, handler &, BinaryOperation combiner,
               const property_list &props = {}) {
  return detail::make_reduction<T, BinaryOperation, false>(vars.data(), 1, combiner,
                                                           nullptr, props);
}
template <typename BufferT, typename BinaryOperation,
          typename = decltype(std::declval<BufferT &>().get_range()),
          typename T = typename BufferT::value_type>
auto reduction(BufferT &vars, handler &, const T &identity, BinaryOperation combiner,
               const property_list &props = {}) {
  return detail::make_reduction<T, BinaryOperation, false>(vars.data(), 1, combiner,
                                                           &identity, props);
}
#if __cplusplus >= 202002L
template <typename T, std::size_t Extent, typename BinaryOperation>
auto reduction(span<T, Extent> vars, BinaryOperation combiner,
               const property_list &props = {}) {
  return detail::make_reduction<T, BinaryOperation, true>(vars.data(), vars.size(),
                                                          combiner, nullptr, props);
}
template <typename T, std::size_t Extent, typename BinaryOperation>
auto reduction(span<T, Extent> vars, const T &identity, BinaryOperation combiner,
               const property_list &props = {}) {
  return detail::make_reduction<T, BinaryOperation, true>(vars.data(), vars.size(),
                                                          combiner, &identity, props);
}
#endif

}

#endif

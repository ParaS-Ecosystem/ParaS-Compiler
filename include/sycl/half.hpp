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

#ifndef __PARAS_HALF_HPP__
#define __PARAS_HALF_HPP__

#include "kem_gpu/gpu_utilities.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace sycl {

class half {
private:
  _Float16 value_;

public:
  PARAS_KERNEL_HD
  constexpr half() noexcept : value_(static_cast<_Float16>(0.0f)) {}

  PARAS_KERNEL_HD
  constexpr half(const half &) noexcept = default;

  PARAS_KERNEL_HD
  constexpr half &operator=(const half &) noexcept = default;

  template <
      typename T,
      typename = std::enable_if_t<std::is_arithmetic_v<T>>>
  PARAS_KERNEL_HD
  constexpr half(T value) noexcept
      : value_(static_cast<_Float16>(value)) {}

  PARAS_KERNEL_HD
  constexpr operator float() const noexcept {
    return static_cast<float>(value_);
  }

  PARAS_KERNEL_HD
  constexpr half &operator+=(half rhs) noexcept {
    value_ =
        static_cast<_Float16>(value_ + rhs.value_);
    return *this;
  }

  PARAS_KERNEL_HD
  constexpr half &operator-=(half rhs) noexcept {
    value_ =
        static_cast<_Float16>(value_ - rhs.value_);
    return *this;
  }

  PARAS_KERNEL_HD
  constexpr half &operator*=(half rhs) noexcept {
    value_ =
        static_cast<_Float16>(value_ * rhs.value_);
    return *this;
  }

  PARAS_KERNEL_HD
  constexpr half &operator/=(half rhs) noexcept {
    value_ =
        static_cast<_Float16>(value_ / rhs.value_);
    return *this;
  }

  PARAS_KERNEL_HD
  constexpr half &operator++() noexcept {
    *this += half(1.0f);
    return *this;
  }

  PARAS_KERNEL_HD
  constexpr half operator++(int) noexcept {
    half old = *this;
    ++(*this);
    return old;
  }

  PARAS_KERNEL_HD
  constexpr half &operator--() noexcept {
    *this -= half(1.0f);
    return *this;
  }

  PARAS_KERNEL_HD
  constexpr half operator--(int) noexcept {
    half old = *this;
    --(*this);
    return old;
  }

  PARAS_KERNEL_HD
  constexpr half operator+() const noexcept {
    return *this;
  }

  PARAS_KERNEL_HD
  constexpr half operator-() const noexcept {
    return half(-static_cast<float>(value_));
  }

  friend PARAS_KERNEL_HD
  constexpr half operator+(half lhs, half rhs) noexcept {
    lhs += rhs;
    return lhs;
  }

  friend PARAS_KERNEL_HD
  constexpr half operator-(half lhs, half rhs) noexcept {
    lhs -= rhs;
    return lhs;
  }

  friend PARAS_KERNEL_HD
  constexpr half operator*(half lhs, half rhs) noexcept {
    lhs *= rhs;
    return lhs;
  }

  friend PARAS_KERNEL_HD
  constexpr half operator/(half lhs, half rhs) noexcept {
    lhs /= rhs;
    return lhs;
  }

  friend PARAS_KERNEL_HD
  constexpr bool operator==(half lhs, half rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend PARAS_KERNEL_HD
  constexpr bool operator!=(half lhs, half rhs) noexcept {
    return lhs.value_ != rhs.value_;
  }

  friend PARAS_KERNEL_HD
  constexpr bool operator<(half lhs, half rhs) noexcept {
    return lhs.value_ < rhs.value_;
  }

  friend PARAS_KERNEL_HD
  constexpr bool operator>(half lhs, half rhs) noexcept {
    return lhs.value_ > rhs.value_;
  }

  friend PARAS_KERNEL_HD
  constexpr bool operator<=(half lhs, half rhs) noexcept {
    return lhs.value_ <= rhs.value_;
  }

  friend PARAS_KERNEL_HD
  constexpr bool operator>=(half lhs, half rhs) noexcept {
    return lhs.value_ >= rhs.value_;
  }
};

static_assert(sizeof(half) == 2,
              "sycl::half must occupy exactly 16 bits");

static_assert(alignof(half) == alignof(_Float16),
              "sycl::half has unexpected alignment");

}

namespace std {

template <>
class numeric_limits<sycl::half> {
public:
  static constexpr bool is_specialized = true;

  static constexpr sycl::half min() noexcept {
    return sycl::half(6.103515625e-05f);
  }

  static constexpr sycl::half max() noexcept {
    return sycl::half(65504.0f);
  }

  static constexpr sycl::half lowest() noexcept {
    return sycl::half(-65504.0f);
  }

  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;

  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;

  static constexpr int radix = 2;

  static constexpr sycl::half epsilon() noexcept {
    return sycl::half(0.0009765625f);
  }

  static constexpr sycl::half round_error() noexcept {
    return sycl::half(0.5f);
  }

  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;

  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;

  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;

  static constexpr float_denorm_style has_denorm =
      denorm_present;

  static constexpr bool has_denorm_loss = false;

  static constexpr sycl::half infinity() noexcept {
    return sycl::half(
        __builtin_inff());
  }

  static constexpr sycl::half quiet_NaN() noexcept {
    return sycl::half(
        __builtin_nanf(""));
  }

  static constexpr sycl::half signaling_NaN() noexcept {
    return sycl::half(
        __builtin_nansf(""));
  }

  static constexpr sycl::half denorm_min() noexcept {
    return sycl::half(5.960464477539063e-08f);
  }

  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;

  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;

  static constexpr float_round_style round_style =
      round_to_nearest;
};

}

#endif

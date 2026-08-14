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

#ifndef __PARAS_DATA_TYPES_HPP__
#define __PARAS_DATA_TYPES_HPP__

#include <cstddef>
#include <cstdint>
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename T, int N> class vec {
public:
  using value_type = T;

  T data[N];

  PARAS_KERNEL_HD
  constexpr vec() noexcept : data{} {}

  PARAS_KERNEL_HD
  explicit constexpr vec(T v) noexcept {
    for (int i = 0; i < N; ++i)
      data[i] = v;
  }

  template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == N>>
  PARAS_KERNEL_HD constexpr vec(Args... args) noexcept
      : data{static_cast<T>(args)...} {}

  PARAS_KERNEL_HD
  static constexpr int size() noexcept { return N; }

  PARAS_KERNEL_HD
  T &operator[](std::size_t i) noexcept { return data[i]; }

  PARAS_KERNEL_HD
  const T &operator[](std::size_t i) const noexcept { return data[i]; }

  PARAS_KERNEL_HD
  vec &operator+=(const vec &rhs) noexcept {
    for (int i = 0; i < N; ++i)
      data[i] += rhs.data[i];
    return *this;
  }

  PARAS_KERNEL_HD
  vec operator+(const vec &rhs) const noexcept {
    vec result;
    for (int i = 0; i < N; ++i)
      result.data[i] = data[i] + rhs.data[i];
    return result;
  }

  PARAS_KERNEL_HD
  vec operator-(const vec &rhs) const noexcept {
    vec result;
    for (int i = 0; i < N; ++i)
      result.data[i] = data[i] - rhs.data[i];
    return result;
  }

  PARAS_KERNEL_HD
  vec &operator*=(T rhs) noexcept {
    for (int i = 0; i < N; ++i) {
      data[i] *= rhs;
    }
    return *this;
  }

  PARAS_KERNEL_HD
  T x() const noexcept {
    static_assert(N >= 1, "x() requires N >= 1");
    return data[0];
  }

  PARAS_KERNEL_HD
  T y() const noexcept {
    static_assert(N >= 2, "y() requires N >= 2");
    return data[1];
  }

  PARAS_KERNEL_HD
  T z() const noexcept {
    static_assert(N >= 3, "z() requires N >= 3");
    return data[2];
  }

  PARAS_KERNEL_HD
  T w() const noexcept {
    static_assert(N >= 4, "w() requires N >= 4");
    return data[3];
  }

  template <typename Ptr>
  PARAS_KERNEL_HD void load(std::size_t offset, Ptr ptr) noexcept {
    const std::size_t base = offset * static_cast<std::size_t>(N);
    for (int j = 0; j < N; ++j) {
      data[j] = ptr[base + static_cast<std::size_t>(j)];
    }
  }

  template <typename Ptr>
  PARAS_KERNEL_HD void store(std::size_t offset, Ptr ptr) const noexcept {
    const std::size_t base = offset * static_cast<std::size_t>(N);
    for (int j = 0; j < N; ++j) {
      ptr[base + static_cast<std::size_t>(j)] = data[j];
    }
  }
  
};

template <typename T, int N>
PARAS_KERNEL_HD constexpr T dot(const vec<T, N> &a,
                                const vec<T, N> &b) noexcept {
  T result{};

  for (int i = 0; i < N; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;

using int2 = vec<std::int32_t, 2>;
using int3 = vec<std::int32_t, 3>;
using int4 = vec<std::int32_t, 4>;

using uint2 = vec<std::uint32_t, 2>;
using uint3 = vec<std::uint32_t, 3>;
using uint4 = vec<std::uint32_t, 4>;

} // namespace sycl

#endif

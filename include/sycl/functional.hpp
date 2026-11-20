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

#ifndef __PARAS_FUNCTIONAL_HPP__
#define __PARAS_FUNCTIONAL_HPP__

#include <utility>
#include <cstddef>
#include <type_traits>

#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename T = void> struct plus {
  PARAS_KERNEL_HD constexpr T operator()(const T &x, const T &y) const {
    return x + y;
  }
};
template <> struct plus<void> {
  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const
      -> decltype(std::forward<T>(x) + std::forward<U>(y)) {
    return std::forward<T>(x) + std::forward<U>(y);
  }
};

template <typename T = void> struct multiplies {
  PARAS_KERNEL_HD constexpr T operator()(const T &x, const T &y) const {
    return x * y;
  }
};
template <> struct multiplies<void> {
  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const
      -> decltype(std::forward<T>(x) * std::forward<U>(y)) {
    return std::forward<T>(x) * std::forward<U>(y);
  }
};

template <typename T = void> struct bit_and {
  PARAS_KERNEL_HD constexpr T operator()(const T &x, const T &y) const {
    return x & y;
  }
};
template <> struct bit_and<void> {
  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const
      -> decltype(std::forward<T>(x) & std::forward<U>(y)) {
    return std::forward<T>(x) & std::forward<U>(y);
  }
};

template <typename T = void> struct bit_or {
  PARAS_KERNEL_HD constexpr T operator()(const T &x, const T &y) const {
    return x | y;
  }
};
template <> struct bit_or<void> {
  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const
      -> decltype(std::forward<T>(x) | std::forward<U>(y)) {
    return std::forward<T>(x) | std::forward<U>(y);
  }
};

template <typename T = void> struct bit_xor {
  PARAS_KERNEL_HD constexpr T operator()(const T &x, const T &y) const {
    return x ^ y;
  }
};
template <> struct bit_xor<void> {
  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const
      -> decltype(std::forward<T>(x) ^ std::forward<U>(y)) {
    return std::forward<T>(x) ^ std::forward<U>(y);
  }
};

template <typename T = void> struct logical_and {
  PARAS_KERNEL_HD constexpr bool operator()(const T &x, const T &y) const {
    return x && y;
  }
};
template <> struct logical_and<void> {
  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const
      -> decltype(std::forward<T>(x) && std::forward<U>(y)) {
    return std::forward<T>(x) && std::forward<U>(y);
  }
};

template <typename T = void> struct logical_or {
  PARAS_KERNEL_HD constexpr bool operator()(const T &x, const T &y) const {
    return x || y;
  }
};
template <> struct logical_or<void> {
  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const
      -> decltype(std::forward<T>(x) || std::forward<U>(y)) {
    return std::forward<T>(x) || std::forward<U>(y);
  }
};

template <typename T = void> struct minimum {
  PARAS_KERNEL_HD constexpr T operator()(const T &x, const T &y) const {
    return (x < y) ? x : y;
  }
};
template <> struct minimum<void> {

  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const {
    using R = std::decay_t<decltype(false ? std::forward<T>(x)
                                          : std::forward<U>(y))>;
    if constexpr (std::is_convertible_v<decltype(x < y), bool>) {
      return R((x < y) ? std::forward<T>(x) : std::forward<U>(y));
    } else {
      R r = x;
      const R o = y;
      for (std::size_t i = 0; i < r.size(); ++i)
        if (!(r[i] < o[i]))
          r[i] = o[i];
      return r;
    }
  }
};

template <typename T = void> struct maximum {
  PARAS_KERNEL_HD constexpr T operator()(const T &x, const T &y) const {
    return (x > y) ? x : y;
  }
};
template <> struct maximum<void> {

  template <typename T, typename U>
  PARAS_KERNEL_HD constexpr auto operator()(T &&x, U &&y) const {
    using R = std::decay_t<decltype(false ? std::forward<T>(x)
                                          : std::forward<U>(y))>;
    if constexpr (std::is_convertible_v<decltype(x > y), bool>) {
      return R((x > y) ? std::forward<T>(x) : std::forward<U>(y));
    } else {
      R r = x;
      const R o = y;
      for (std::size_t i = 0; i < r.size(); ++i)
        if (!(r[i] > o[i]))
          r[i] = o[i];
      return r;
    }
  }
};

}

#endif

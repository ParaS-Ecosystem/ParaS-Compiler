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

#ifndef __PARAS_MATH_HPP__
#define __PARAS_MATH_HPP__

#include <type_traits>
#include <cmath>
#include <algorithm>
#include "kem_gpu/gpu_utilities.hpp"
#include <utility>
#include "data_types.hpp"

namespace sycl {

template <typename T, std::size_t N>
class marray;
template <typename T, int N>
class vec;

template <typename T>
struct is_marray : std::false_type {};
template <typename T, std::size_t N>
struct is_marray<marray<T, N>> : std::true_type {};
template <typename T>
inline constexpr bool is_marray_v = is_marray<T>::value;

template <typename T>
struct is_vec : std::false_type {};
template <typename T, int N>
struct is_vec<vec<T, N>> : std::true_type {};
template <typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

template <typename T>
struct vector_size;
template <typename T, std::size_t N>
struct vector_size<marray<T, N>> : std::integral_constant<std::size_t, N> {};
template <typename T, int N>
struct vector_size<vec<T, N>> : std::integral_constant<std::size_t, static_cast<std::size_t>(N)> {};

}
#include "builtins.hpp"
namespace sycl {

PARAS_KERNEL_HD
inline float logf(float x) {
    return ::logf(x);
}

PARAS_KERNEL_HD
inline double dot(double* a, double* b, size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

template <typename T, typename U,
          std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U> &&
                               !std::is_same_v<T, U>,
                           int> = 0>
PARAS_KERNEL_HD inline auto max(T a, U b) {
    using type = std::common_type_t<T, U>;
    const type x = static_cast<type>(a);
    const type y = static_cast<type>(b);
    return (x > y) ? x : y;
}

template <typename T, typename U,
          std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U> &&
                               !std::is_same_v<T, U>,
                           int> = 0>
PARAS_KERNEL_HD inline auto min(T a, U b) {
    using type = std::common_type_t<T, U>;
    const type x = static_cast<type>(a);
    const type y = static_cast<type>(b);
    return (x < y) ? x : y;
}

/* ============================== SYCL Functional Objects ============================== */

template <typename T = void>
struct plus {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T = void>
struct minimum {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return (a < b) ? a : b; }
};

template <typename T = void>
struct maximum {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return (a > b) ? a : b; }
};

template <typename T = void>
struct multiplies {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return a * b; }
};

template <typename T = void>
struct bit_and {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return a & b; }
};

template <typename T = void>
struct bit_or {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return a | b; }
};

template <typename T = void>
struct bit_xor {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return a ^ b; }
};

template <typename T = void>
struct logical_and {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return a && b; }
};

template <typename T = void>
struct logical_or {
    PARAS_KERNEL_HD
    constexpr T operator()(const T& a, const T& b) const { return a || b; }
};

template <>
struct plus<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(std::forward<T>(a) + std::forward<U>(b)) {
        return std::forward<T>(a) + std::forward<U>(b);
    }
};

template <>
struct minimum<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const {
        using DT = std::decay_t<T>;
        if constexpr (is_marray_v<DT> || is_vec_v<DT>) {
            DT result{};
            constexpr std::size_t N = vector_size<DT>::value;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = (a[i] < b[i]) ? a[i] : b[i];
            }
            return result;
        } else {
            return std::forward<T>(a) < std::forward<U>(b) ? std::forward<T>(a)
                                                           : std::forward<U>(b);
        }
    }
};

template <>
struct maximum<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const {
        using DT = std::decay_t<T>;
        if constexpr (is_marray_v<DT> || is_vec_v<DT>) {
            DT result{};
            constexpr std::size_t N = vector_size<DT>::value;
            for (std::size_t i = 0; i < N; ++i) {
                result[i] = (a[i] > b[i]) ? a[i] : b[i];
            }
            return result;
        } else {
            return std::forward<T>(a) > std::forward<U>(b) ? std::forward<T>(a)
                                                           : std::forward<U>(b);
        }
    }
};

template <>
struct multiplies<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(std::forward<T>(a) * std::forward<U>(b)) {
        return std::forward<T>(a) * std::forward<U>(b);
    }
};

template <>
struct bit_and<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(std::forward<T>(a) & std::forward<U>(b)) {
        return std::forward<T>(a) & std::forward<U>(b);
    }
};

template <>
struct bit_or<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(std::forward<T>(a) | std::forward<U>(b)) {
        return std::forward<T>(a) | std::forward<U>(b);
    }
};

template <>
struct bit_xor<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(std::forward<T>(a) ^ std::forward<U>(b)) {
        return std::forward<T>(a) ^ std::forward<U>(b);
    }
};

template <>
struct logical_and<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(std::forward<T>(a) && std::forward<U>(b)) {
        return std::forward<T>(a) && std::forward<U>(b);
    }
};

template <>
struct logical_or<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(std::forward<T>(a) || std::forward<U>(b)) {
        return std::forward<T>(a) || std::forward<U>(b);
    }
};

template <typename T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                                           std::is_same_v<std::remove_cv_t<T>, double>,
                                       int> = 0>
PARAS_KERNEL_HD inline T nearbyint(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return ::nearbyintf(x);
    } else {
        return ::nearbyint(x);
    }
}

template <typename T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                                           std::is_same_v<std::remove_cv_t<T>, double>,
                                       int> = 0>
PARAS_KERNEL_HD inline T llrint(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return ::llrintf(x);
    } else {
        return ::llrint(x);
    }
}

template <typename T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                                           std::is_same_v<std::remove_cv_t<T>, double>,
                                       int> = 0>
PARAS_KERNEL_HD inline T llround(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return ::llroundf(x);
    } else {
        return ::llround(x);
    }
}

template <typename T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                                           std::is_same_v<std::remove_cv_t<T>, double>,
                                       int> = 0>
PARAS_KERNEL_HD inline T lrint(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return ::lrintf(x);
    } else {
        return ::lrint(x);
    }
}

template <typename T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                                           std::is_same_v<std::remove_cv_t<T>, double>,
                                       int> = 0>
PARAS_KERNEL_HD inline T lround(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return ::lroundf(x);
    } else {
        return ::lround(x);
    }
}

} // namespace sycl

#endif

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

PARAS_KERNEL_HD
inline double sin(double x) {
    return ::sin(x);
}

PARAS_KERNEL_HD
inline float sin(float x) {
    return std::sin(x);
}

PARAS_KERNEL_HD
inline sycl::half sin(sycl::half x) {
    return sycl::half(std::sin(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto sin(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::sin(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double cos(double x) {
    return ::cos(x);
}

PARAS_KERNEL_HD
inline float cos(float x) {
    return std::cos(x);
}

PARAS_KERNEL_HD
inline sycl::half cos(sycl::half x) {
    return sycl::half(std::cos(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto cos(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::cos(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double erf(double x) {
    return ::erf(x);
}

PARAS_KERNEL_HD
inline float erf(float x) {
    return std::erf(x);
}

PARAS_KERNEL_HD
inline sycl::half erf(sycl::half x) {
    return sycl::half(std::erf(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto erf(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::erf(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double tan(double x) {
    return ::tan(x);
}

PARAS_KERNEL_HD
inline float tan(float x) {
    return std::tan(x);
}

PARAS_KERNEL_HD
inline sycl::half tan(sycl::half x) {
    return sycl::half(std::tan(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto tan(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::tan(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double sqrt(double x) {
    return ::sqrt(x);
}

PARAS_KERNEL_HD
inline float sqrt(float x) {
    return ::sqrtf(x);
}

PARAS_KERNEL_HD
inline sycl::half sqrt(sycl::half x) {
    return sycl::half(std::sqrt(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto sqrt(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::sqrt(x[i]);
        }
        return result;
    }
}

#if PARAS_GPU_BACKEND
PARAS_KERNEL_HD
inline double rsqrt(double x) {
    return ::rsqrt(x);
}

PARAS_KERNEL_HD
inline float rsqrt(float x) {
    return ::rsqrtf(x);
}
#else
inline double rsqrt(double x) {
    return 1.0 / std::sqrt(x);
}

inline float rsqrt(float x) {
    return 1.0f / std::sqrt(x);
}
#endif

PARAS_KERNEL_HD
inline double dot(double* a, double* b, size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

PARAS_KERNEL_HD
inline double exp(double x) {
    return ::exp(x);
}

PARAS_KERNEL_HD
inline float exp(float x) {
    return std::exp(x);
}

PARAS_KERNEL_HD
inline sycl::half exp(sycl::half x) {
    return sycl::half(std::exp(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto exp(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::exp(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double acos(double x) {
    return ::acos(x);
}

PARAS_KERNEL_HD
inline float acos(float x) {
    return std::acos(x);
}

PARAS_KERNEL_HD
inline sycl::half acos(sycl::half x) {
    return sycl::half(std::acos(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto acos(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::acos(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double atan(double x) {
    return ::atan(x);
}

PARAS_KERNEL_HD
inline float atan(float x) {
    return std::atan(x);
}

PARAS_KERNEL_HD
inline sycl::half atan(sycl::half x) {
    return sycl::half(std::atan(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto atan(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::atan(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double atan2(double y, double x) {
    return ::atan2(y, x);
}

PARAS_KERNEL_HD
inline float atan2(float y, float x) {
    return std::atan2(y, x);
}

PARAS_KERNEL_HD
inline sycl::half atan2(sycl::half y, sycl::half x) {
    return sycl::half(std::atan2(static_cast<float>(y), static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto atan2(const NonScalar& y, const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < y.size(); ++i) {
            result[i] = sycl::atan2(y[i], x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double asin(double x) {
    return ::asin(x);
}

PARAS_KERNEL_HD
inline float asin(float x) {
    return std::asin(x);
}

PARAS_KERNEL_HD
inline sycl::half asin(sycl::half x) {
    return sycl::half(std::asin(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto asin(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::asin(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double sinh(double x) {
    return ::sinh(x);
}

PARAS_KERNEL_HD
inline float sinh(float x) {
    return std::sinh(x);
}

PARAS_KERNEL_HD
inline sycl::half sinh(sycl::half x) {
    return sycl::half(std::sinh(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto sinh(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::sinh(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double asinh(double x) {
    return ::asinh(x);
}

PARAS_KERNEL_HD
inline float asinh(float x) {
    return std::asinh(x);
}

PARAS_KERNEL_HD
inline sycl::half asinh(sycl::half x) {
    return sycl::half(std::asinh(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto asinh(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::asinh(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double acosh(double x) {
    return ::acosh(x);
}

PARAS_KERNEL_HD
inline double cosh(double x) {
    return ::cosh(x);
}

PARAS_KERNEL_HD
inline double rint(double x) {
    return ::rint(x);
}

PARAS_KERNEL_HD
inline float cosh(float x) {
    return std::cosh(x);
}

PARAS_KERNEL_HD
inline sycl::half cosh(sycl::half x) {
    return sycl::half(std::cosh(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto cosh(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::cosh(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float acosh(float x) {
    return std::acosh(x);
}

PARAS_KERNEL_HD
inline sycl::half acosh(sycl::half x) {
    return sycl::half(std::acosh(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto acosh(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::acosh(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float rint(float x) {
    return ::rintf(x);
}

PARAS_KERNEL_HD
inline sycl::half rint(sycl::half x) {
    return sycl::half(std::rint(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto rint(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::rint(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline bool isfinite(double x) {
    return std::isfinite(x);
}

PARAS_KERNEL_HD
inline bool isfinite(float x) {
    return std::isfinite(x);
}

PARAS_KERNEL_HD
inline bool isfinite(sycl::half x) {
    return std::isfinite(static_cast<float>(x));
}

template <typename NonScalar, std::enable_if_t<is_vec_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto isfinite(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar>) {
        using T = typename NonScalar::element_type;
        vec<typename vec_rel_t<T>::type, vector_size<NonScalar>::value> result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::isfinite(x[i]) ? typename vec_rel_t<T>::type(-1)
                                             : typename vec_rel_t<T>::type(0);
        }
        return result;
    } else if constexpr (is_marray_v<NonScalar>) {
        marray<bool, vector_size<NonScalar>::value> result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::isfinite(x[i]);
        }
        return result;
    }
}

template <typename T>
PARAS_KERNEL_HD inline T tanh(T x) {
#if PARAS_GPU_BACKEND
    return ::tanh(x);
#else
    using std::tanh;
    return tanh(x);
#endif
}

PARAS_KERNEL_HD
inline double fma(double x, double y, double z) {
    return ::fma(x, y, z);
}

PARAS_KERNEL_HD
inline float fma(float x, float y, float z) {
    return std::fma(x, y, z);
}

PARAS_KERNEL_HD
inline sycl::half fma(sycl::half x, sycl::half y, sycl::half z) {
    return sycl::half(
        std::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto fma(const NonScalar& x, const NonScalar& y, const NonScalar& z) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::fma(x[i], y[i], z[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double fmax(double x, double y) {
    return ::fmax(x, y);
}

PARAS_KERNEL_HD
inline float fmax(float x, float y) {
    return std::fmax(x, y);
}

PARAS_KERNEL_HD
inline sycl::half fmax(sycl::half x, sycl::half y) {
    return sycl::half(std::fmax(static_cast<float>(x), static_cast<float>(y)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto fmax(const NonScalar& x, const NonScalar& y) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::fmax(x[i], y[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double fmin(double x, double y) {
    return ::fmin(x, y);
}

PARAS_KERNEL_HD
inline float fmin(float x, float y) {
    return std::fmin(x, y);
}

PARAS_KERNEL_HD
inline sycl::half fmin(sycl::half x, sycl::half y) {
    return sycl::half(std::fmin(static_cast<float>(x), static_cast<float>(y)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto fmin(const NonScalar& x, const NonScalar& y) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::fmin(x[i], y[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double fdim(double x, double y) {
    return ::fdim(x, y);
}

PARAS_KERNEL_HD
inline float fdim(float x, float y) {
    return std::fdim(x, y);
}

PARAS_KERNEL_HD
inline sycl::half fdim(sycl::half x, sycl::half y) {
    return sycl::half(std::fdim(static_cast<float>(x), static_cast<float>(y)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto fdim(const NonScalar& x, const NonScalar& y) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::fdim(x[i], y[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline int popcount(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }

    return count;
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto popcount(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = static_cast<typename NonScalar::value_type>(
                sycl::popcount(static_cast<unsigned int>(x[i])));
        }
        return result;
    }
}

template <typename T, typename U>
PARAS_KERNEL_HD inline auto max(T a, U b) {
    using type = std::common_type_t<T, U>;
    const type x = static_cast<type>(a);
    const type y = static_cast<type>(b);
    return (x > y) ? x : y;
}

template <typename T, typename U>
PARAS_KERNEL_HD inline auto min(T a, U b) {
    using type = std::common_type_t<T, U>;
    const type x = static_cast<type>(a);
    const type y = static_cast<type>(b);
    return (x < y) ? x : y;
}

/* ===================== LOG FUNCTIONS ===================== */

PARAS_KERNEL_HD
inline float logf(float x) {
    return ::logf(x);
}

PARAS_KERNEL_HD
inline double log(double x) {
    return ::log(x);
}

PARAS_KERNEL_HD
inline float log(float x) {
    return std::log(x);
}

PARAS_KERNEL_HD
inline sycl::half log(sycl::half x) {
    return sycl::half(std::log(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto log(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::log(x[i]);
        }
        return result;
    }
}

template <typename T>
PARAS_KERNEL_HD inline T log10(T x) {
#if PARAS_GPU_BACKEND
    return ::log10(x);
#else
    using std::log10;
    return log10(x);
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T log1p(T x) {
#if PARAS_GPU_BACKEND
    return ::log1p(x);
#else
    using std::log1p;
    return log1p(x);
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T log2(T x) {
#if PARAS_GPU_BACKEND
    return ::log2(x);
#else
    using std::log2;
    return log2(x);
#endif
}

template <typename T>
PARAS_KERNEL_HD inline T logb(T x) {
#if PARAS_GPU_BACKEND
    return ::logb(x);
#else
    using std::logb;
    return logb(x);
#endif
}

/* ===================== LOG FUNCTIONS ===================== */

/* ===================== POW FUNCTIONS ===================== */

PARAS_KERNEL_HD
inline double exp2(double x) {
    return ::exp2(x);
}

PARAS_KERNEL_HD
inline sycl::half exp2(sycl::half x) {
    return sycl::half(std::exp2(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto exp2(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::exp2(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float exp2(float x) {
    return ::exp2f(x);
}

PARAS_KERNEL_HD
inline double exp10(double x) {
#if PARAS_GPU_BACKEND
    return ::exp10(x);
#else
    return ::pow(10.0, x);
#endif
}

PARAS_KERNEL_HD
inline float exp10(float x) {
#if PARAS_GPU_BACKEND
    return ::exp10f(x);
#else
    return ::powf(10.0f, x);
#endif
}

PARAS_KERNEL_HD
inline sycl::half exp10(sycl::half x) {
    return sycl::half(sycl::exp10(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto exp10(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::exp10(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline double expm1(double x) {
    return ::expm1(x);
}

PARAS_KERNEL_HD
inline sycl::half expm1(sycl::half x) {
    return sycl::half(std::expm1(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto expm1(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::expm1(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float expm1(float x) {
    return ::expm1f(x);
}

PARAS_KERNEL_HD
inline double pow(double x, double y) {
    return ::pow(x, y);
}

PARAS_KERNEL_HD
inline sycl::half pow(sycl::half x, sycl::half y) {
    return sycl::half(std::pow(static_cast<float>(x), static_cast<float>(y)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto pow(const NonScalar& x, const NonScalar& y) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::pow(x[i], y[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float pow(float x, float y) {
    return ::powf(x, y);
}

/* ===================== POW FUNCTIONS ===================== */

/* ============= FABS / NEXTAFTER / ROUNDING FUNCTIONS ============= */

PARAS_KERNEL_HD
inline double fabs(double x) {
    return ::fabs(x);
}

PARAS_KERNEL_HD
inline sycl::half fabs(sycl::half x) {
    return sycl::half(std::fabs(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto fabs(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::fabs(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float fabs(float x) {
    return ::fabsf(x);
}

PARAS_KERNEL_HD
inline double nextafter(double x, double y) {
    return ::nextafter(x, y);
}

PARAS_KERNEL_HD
inline sycl::half nextafter(sycl::half x, sycl::half y) {
    return sycl::half(std::nextafter(static_cast<float>(x), static_cast<float>(y)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto nextafter(const NonScalar& x, const NonScalar& y) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::nextafter(x[i], y[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float nextafter(float x, float y) {
    return ::nextafterf(x, y);
}

PARAS_KERNEL_HD
inline double trunc(double x) {
    return ::trunc(x);
}

PARAS_KERNEL_HD
inline sycl::half trunc(sycl::half x) {
    return sycl::half(std::trunc(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto trunc(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::trunc(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float trunc(float x) {
    return ::truncf(x);
}

PARAS_KERNEL_HD
inline double ceil(double x) {
    return ::ceil(x);
}

PARAS_KERNEL_HD
inline sycl::half ceil(sycl::half x) {
    return sycl::half(std::ceil(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto ceil(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::ceil(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float ceil(float x) {
    return ::ceilf(x);
}

PARAS_KERNEL_HD
inline double floor(double x) {
    return ::floor(x);
}

PARAS_KERNEL_HD
inline sycl::half floor(sycl::half x) {
    return sycl::half(std::floor(static_cast<float>(x)));
}

template <typename NonScalar,
          std::enable_if_t<is_vec_v<NonScalar> || is_marray_v<NonScalar>, int> = 0>
PARAS_KERNEL_HD inline auto floor(const NonScalar& x) {
    if constexpr (is_vec_v<NonScalar> || is_marray_v<NonScalar>) {
        NonScalar result{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            result[i] = sycl::floor(x[i]);
        }
        return result;
    }
}

PARAS_KERNEL_HD
inline float floor(float x) {
    return ::floorf(x);
}

template <typename T, int N>
PARAS_KERNEL_HD inline vec<T, N> fabs(const vec<T, N>& v) {
    vec<T, N> result;
    for (int i = 0; i < N; ++i)
        result[i] = fabs(v[i]);
    return result;
}

template <typename T, int N>
PARAS_KERNEL_HD inline vec<T, N> trunc(const vec<T, N>& v) {
    vec<T, N> result;
    for (int i = 0; i < N; ++i)
        result[i] = trunc(v[i]);
    return result;
}

template <typename T, int N>
PARAS_KERNEL_HD inline vec<T, N> ceil(const vec<T, N>& v) {
    vec<T, N> result;
    for (int i = 0; i < N; ++i)
        result[i] = ceil(v[i]);
    return result;
}

template <typename T, int N>
PARAS_KERNEL_HD inline vec<T, N> floor(const vec<T, N>& v) {
    vec<T, N> result;
    for (int i = 0; i < N; ++i)
        result[i] = floor(v[i]);
    return result;
}

template <typename T, int N>
PARAS_KERNEL_HD inline vec<T, N> rint(const vec<T, N>& v) {
    vec<T, N> result;
    for (int i = 0; i < N; ++i)
        result[i] = rint(v[i]);
    return result;
}

/* ============= FABS / NEXTAFTER / ROUNDING FUNCTIONS ============= */

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

template <typename T>
PARAS_KERNEL_HD inline T pown(T x, int y) {
    return pow(x, static_cast<T>(y));
}

PARAS_KERNEL_HD
inline float ldexp(float x, int k) {
    return ::ldexpf(x, k);
}

PARAS_KERNEL_HD
inline double ldexp(double x, int k) {
    return ::ldexp(x, k);
}

PARAS_KERNEL_HD
inline float frexp(float x, int* exp) {
    return ::frexpf(x, exp);
}

PARAS_KERNEL_HD
inline double frexp(double x, int* exp) {
    return ::frexp(x, exp);
}

template <typename T, std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, float> ||
                                           std::is_same_v<std::remove_cv_t<T>, double>,
                                       int> = 0>
PARAS_KERNEL_HD inline T round(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return ::roundf(x);
    } else {
        return ::round(x);
    }
}

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

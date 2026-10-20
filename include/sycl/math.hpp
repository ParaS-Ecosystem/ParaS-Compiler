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

#include <cmath>
#include <algorithm>
#include "kem_gpu/gpu_utilities.hpp"
#include "functional.hpp"
#include <utility>

#include "data_types.hpp"

namespace sycl {

template <typename T, std::size_t N> class marray;
template <typename T, int N> class vec;

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
struct vector_size<vec<T, N>>
    : std::integral_constant<std::size_t, static_cast<std::size_t>(N)> {};

    PARAS_KERNEL_HD
    inline double sqrt(double x) {
	    return ::sqrt(x) ;
    }

    PARAS_KERNEL_HD
    inline float sqrt(float x) {
        return ::sqrtf(x);
    }

    #if PARAS_GPU_BACKEND
        PARAS_KERNEL_HD
        inline double rsqrt(double x) {
        #if PARAS_DEVICE_PASS
            return ::rsqrt(x) ;
        #else
            return 1.0 / std::sqrt(x) ;
        #endif
        }

        PARAS_KERNEL_HD
        inline float rsqrt(float x) {
        #if PARAS_DEVICE_PASS
            return ::rsqrtf(x);
        #else
            return 1.0f / std::sqrt(x);
        #endif
        }
    #else
        PARAS_KERNEL_HD inline double rsqrt(double x) {
            return 1.0 / std::sqrt(x) ;
        }

        PARAS_KERNEL_HD inline float rsqrt(float x) {
            return 1.0f / std::sqrt(x);
        }
    #endif

    PARAS_KERNEL_HD
    inline double dot(double* a, double*b, size_t n) {
        double result = 0.0 ;
        for (size_t i = 0 ; i < n ; ++i) {
            result += a[i] * b[i] ;
        }
        return result ;
    }

    PARAS_KERNEL_HD
    inline double atan2(double y, double x) {
        return ::atan2(y, x) ;
    }

    PARAS_KERNEL_HD
    inline double rint(double x) {
        return ::rint(x) ;
    }

    PARAS_KERNEL_HD
    inline float rint(float x) {
        return ::rintf(x);
    }

    PARAS_KERNEL_HD
    inline double fma(double x, double y, double z) {
        return ::fma(x, y, z) ;
    }

    PARAS_KERNEL_HD
    inline double fmax(double x, double y) {
        return ::fmax(x, y) ;
    }

    PARAS_KERNEL_HD
    inline double fmin(double x, double y) {
        return ::fmin(x, y) ;
    }

    PARAS_KERNEL_HD
    inline double fdim(double x, double y) {
        return ::fdim(x, y) ;
    }

    PARAS_KERNEL_HD
    inline float logf(float x) {
        return ::logf(x) ;
    }

PARAS_KERNEL_HD
inline double exp2(double x) {
    return ::exp2(x);
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
inline double expm1(double x) {
    return ::expm1(x);
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
inline float pow(float x, float y) {
    return ::powf(x, y);
}

PARAS_KERNEL_HD
inline double fabs(double x) {
    return ::fabs(x);
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
inline float nextafter(float x, float y) {
    return ::nextafterf(x, y);
}

PARAS_KERNEL_HD
inline double trunc(double x) {
    return ::trunc(x);
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
inline float ceil(float x) {
    return ::ceilf(x);
}

PARAS_KERNEL_HD
inline double floor(double x) {
    return ::floor(x);
}

PARAS_KERNEL_HD
inline float floor(float x) {
    return ::floorf(x);
}

}

#endif

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

#include <limits>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

using byte = unsigned char;

enum class elem : int {
    x = 0,
    y = 1,
    z = 2,
    w = 3,
    r = 0,
    g = 1,
    b = 2,
    a = 3,
    s0 = 0,
    s1 = 1,
    s2 = 2,
    s3 = 3,
    s4 = 4,
    s5 = 5,
    s6 = 6,
    s7 = 7,
    s8 = 8,
    s9 = 9,
    sA = 10,
    sB = 11,
    sC = 12,
    sD = 13,
    sE = 14,
    sF = 15
};

enum class rounding_mode { automatic, rte, rtz, rtp, rtn };

template <typename T>
struct vec_rel_t;

template <typename T, typename = void>
struct paras_is_swizzle_ref;

template <typename T, int N>
constexpr std::size_t paras_vec_alignof() {
    std::size_t count = (N == 3) ? 4 : static_cast<std::size_t>(N);
    std::size_t bytes = sizeof(T) * count;
    return (bytes > 64) ? 64 : bytes;
}
template <typename T, int N>

class alignas(paras_vec_alignof<T, N>()) vec {
public:
    using value_type = T;
    using element_type = T;
    T data[N];

    PARAS_KERNEL_HD
    constexpr vec() noexcept : data{} {}

    PARAS_KERNEL_HD
    explicit constexpr vec(T v) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] = v;
    }

    PARAS_KERNEL_HD
    vec& operator=(const T& v) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] = v;
        return *this;
    }

    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator=(const SwizzleRefT& rhs) noexcept {
        return (*this) = static_cast<vec<T, N>>(rhs);
    }

private:
    template <typename U>
    PARAS_KERNEL_HD static constexpr int paras_fill_component(T* dst, int idx,
                                                              const U& val) noexcept {
        if constexpr (std::is_convertible_v<U, T>) {
            dst[idx] = static_cast<T>(val);
            return idx + 1;
        } else {
            for (std::size_t j = 0; j < static_cast<std::size_t>(U::size()); ++j)
                dst[idx + static_cast<int>(j)] = val[j];
            return idx + static_cast<int>(U::size());
        }
    }

public:
    template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) >= 1)>>
    PARAS_KERNEL_HD explicit constexpr vec(Args... args) noexcept : data{} {
        int idx = 0;
        ((idx = paras_fill_component(data, idx, args)), ...);
    }

    PARAS_KERNEL_HD
    static constexpr int size() noexcept { return N; }

    PARAS_KERNEL_HD
    std::size_t byte_size() const noexcept {
        constexpr std::size_t count = (N == 3) ? 4 : static_cast<std::size_t>(N);
        return sizeof(T) * count;
    }

    PARAS_KERNEL_HD
    T& operator[](int i) noexcept { return data[i]; }

    PARAS_KERNEL_HD
    const T& operator[](int i) const noexcept { return data[i]; }

    PARAS_KERNEL_HD
    vec& operator+=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] += rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator+=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] += rhs;
        return *this;
    }

    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator+=(const SwizzleRefT& rhs) noexcept {
        return (*this) += static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator+(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] + rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec operator-(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] - rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec operator-(T rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] - rhs;
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator*=(T rhs) noexcept {
        for (int i = 0; i < N; ++i) {
            data[i] *= rhs;
        }
        return *this;
    }
    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator*=(const SwizzleRefT& rhs) noexcept {
        return (*this) *= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec& operator-=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] -= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator-=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] -= rhs;
        return *this;
    }

    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator-=(const SwizzleRefT& rhs) noexcept {
        return (*this) -= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec& operator*=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] *= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec operator*(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] * rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator/=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] /= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator/=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] /= rhs;
        return *this;
    }
    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator/=(const SwizzleRefT& rhs) noexcept {
        return (*this) /= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator/(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] / rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec operator/(T rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] / rhs;
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator%=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] %= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator%=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] %= rhs;
        return *this;
    }
    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator%=(const SwizzleRefT& rhs) noexcept {
        return (*this) %= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator%(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] % rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec operator%(T rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] % rhs;
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator&=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] &= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator&=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] &= rhs;
        return *this;
    }

    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator&=(const SwizzleRefT& rhs) noexcept {
        return (*this) &= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator&(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] & rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator|=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] |= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator|=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] |= rhs;
        return *this;
    }
    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator|=(const SwizzleRefT& rhs) noexcept {
        return (*this) |= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator|(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] | rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator^=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] ^= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator^=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] ^= rhs;
        return *this;
    }

    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator^=(const SwizzleRefT& rhs) noexcept {
        return (*this) ^= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator^(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] ^ rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator<<=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] <<= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator<<=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] <<= rhs;
        return *this;
    }

    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator<<=(const SwizzleRefT& rhs) noexcept {
        return (*this) <<= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator<<(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] << rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec operator<<(T rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] << rhs;
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator>>=(const vec& rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] >>= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec& operator>>=(T rhs) noexcept {
        for (int i = 0; i < N; ++i)
            data[i] >>= rhs;
        return *this;
    }

    template <
        typename SwizzleRefT,
        typename = std::enable_if_t<static_cast<int>(SwizzleRefT::size()) == N &&
                                    std::is_convertible_v<typename SwizzleRefT::element_type, T>>>
    PARAS_KERNEL_HD vec& operator>>=(const SwizzleRefT& rhs) noexcept {
        return (*this) >>= static_cast<vec>(rhs);
    }

    PARAS_KERNEL_HD
    vec operator>>(const vec& rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] >> rhs.data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec operator>>(T rhs) const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = data[i] >> rhs;
        return result;
    }

    PARAS_KERNEL_HD
    vec operator+() const noexcept { return *this; }

    PARAS_KERNEL_HD
    vec operator-() const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = -data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec operator~() const noexcept {
        vec result;
        for (int i = 0; i < N; ++i)
            result.data[i] = ~data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec<typename vec_rel_t<T>::type, N> operator!() const noexcept {
        using RelT = typename vec_rel_t<T>::type;
        vec<RelT, N> result{};
        for (int i = 0; i < N; ++i)
            result[i] = (!static_cast<bool>(data[i])) ? RelT(-1) : RelT(0);
        return result;
    }

    PARAS_KERNEL_HD
    vec& operator++() noexcept {
        for (int i = 0; i < N; ++i)
            ++data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec operator++(int) noexcept {
        vec tmp(*this);
        ++(*this);
        return tmp;
    }

    PARAS_KERNEL_HD
    vec& operator--() noexcept {
        for (int i = 0; i < N; ++i)
            --data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    vec operator--(int) noexcept {
        vec tmp(*this);
        --(*this);
        return tmp;
    }

    PARAS_KERNEL_HD
    T x() const noexcept {
        static_assert(N >= 1, "x() requires N >= 1");
        return data[0];
    }

    PARAS_KERNEL_HD
    T& x() noexcept {
        static_assert(N >= 1, "x() requires N >= 1");
        return data[0];
    }

    PARAS_KERNEL_HD
    T y() const noexcept {
        static_assert(N >= 2, "y() requires N >= 2");
        return data[1];
    }

    PARAS_KERNEL_HD
    T& y() noexcept {
        static_assert(N >= 2, "y() requires N >= 2");
        return data[1];
    }

    PARAS_KERNEL_HD
    T z() const noexcept {
        static_assert(N >= 3, "z() requires N >= 3");
        return data[2];
    }

    PARAS_KERNEL_HD
    T& z() noexcept {
        static_assert(N >= 3, "z() requires N >= 3");
        return data[2];
    }

    PARAS_KERNEL_HD
    T w() const noexcept {
        static_assert(N >= 4, "w() requires N >= 4");
        return data[3];
    }

    PARAS_KERNEL_HD
    T& w() noexcept {
        static_assert(N >= 4, "w() requires N >= 4");
        return data[3];
    }

#define PARAS_VEC_SWIZZLE_ACCESSOR(NAME, INDEX)                                                    \
    PARAS_KERNEL_HD                                                                                \
    T NAME() const noexcept {                                                                      \
        static_assert(N > (INDEX), #NAME "() requires N > " #INDEX);                               \
        return data[INDEX];                                                                        \
    }                                                                                              \
    PARAS_KERNEL_HD                                                                                \
    T& NAME() noexcept {                                                                           \
        static_assert(N > (INDEX), #NAME "() requires N > " #INDEX);                               \
        return data[INDEX];                                                                        \
    }

    PARAS_VEC_SWIZZLE_ACCESSOR(s0, 0)
    PARAS_VEC_SWIZZLE_ACCESSOR(s1, 1)
    PARAS_VEC_SWIZZLE_ACCESSOR(s2, 2)
    PARAS_VEC_SWIZZLE_ACCESSOR(s3, 3)
    PARAS_VEC_SWIZZLE_ACCESSOR(s4, 4)
    PARAS_VEC_SWIZZLE_ACCESSOR(s5, 5)
    PARAS_VEC_SWIZZLE_ACCESSOR(s6, 6)
    PARAS_VEC_SWIZZLE_ACCESSOR(s7, 7)
    PARAS_VEC_SWIZZLE_ACCESSOR(s8, 8)
    PARAS_VEC_SWIZZLE_ACCESSOR(s9, 9)
    PARAS_VEC_SWIZZLE_ACCESSOR(sA, 10)
    PARAS_VEC_SWIZZLE_ACCESSOR(sB, 11)
    PARAS_VEC_SWIZZLE_ACCESSOR(sC, 12)
    PARAS_VEC_SWIZZLE_ACCESSOR(sD, 13)
    PARAS_VEC_SWIZZLE_ACCESSOR(sE, 14)
    PARAS_VEC_SWIZZLE_ACCESSOR(sF, 15)

#undef PARAS_VEC_SWIZZLE_ACCESSOR

#define PARAS_VEC_NAMED_SWIZZLE2(NAME, I0, I1)                                                     \
    PARAS_KERNEL_HD                                                                                \
    vec<T, 2> NAME() const noexcept {                                                              \
        static_assert(N > (I0) && N > (I1), #NAME "() index out of range");                        \
        return vec<T, 2>{data[I0], data[I1]};                                                      \
    }

#define PARAS_VEC_NAMED_SWIZZLE3(NAME, I0, I1, I2)                                                 \
    PARAS_KERNEL_HD                                                                                \
    vec<T, 3> NAME() const noexcept {                                                              \
        static_assert(N > (I0) && N > (I1) && N > (I2), #NAME "() index out of range");            \
        return vec<T, 3>{data[I0], data[I1], data[I2]};                                            \
    }

#define PARAS_VEC_NAMED_SWIZZLE4(NAME, I0, I1, I2, I3)                                             \
    PARAS_KERNEL_HD                                                                                \
    vec<T, 4> NAME() const noexcept {                                                              \
        static_assert(N > (I0) && N > (I1) && N > (I2) && N > (I3),                                \
                      #NAME "() index out of range");                                              \
        return vec<T, 4>{data[I0], data[I1], data[I2], data[I3]};                                  \
    }

    PARAS_VEC_NAMED_SWIZZLE2(xx, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE2(xy, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE2(xz, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE2(xw, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE2(yx, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE2(yy, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE2(yz, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE2(yw, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE2(zx, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE2(zy, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE2(zz, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE2(zw, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE2(wx, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE2(wy, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE2(wz, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE2(ww, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(xxx, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(xxy, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(xxz, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(xxw, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(xyx, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(xyy, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(xyz, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(xyw, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(xzx, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(xzy, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(xzz, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(xzw, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(xwx, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(xwy, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(xwz, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(xww, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(yxx, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(yxy, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(yxz, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(yxw, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(yyx, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(yyy, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(yyz, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(yyw, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(yzx, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(yzy, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(yzz, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(yzw, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(ywx, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(ywy, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(ywz, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(yww, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(zxx, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(zxy, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(zxz, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(zxw, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(zyx, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(zyy, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(zyz, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(zyw, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(zzx, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(zzy, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(zzz, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(zzw, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(zwx, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(zwy, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(zwz, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(zww, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(wxx, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(wxy, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(wxz, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(wxw, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(wyx, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(wyy, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(wyz, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(wyw, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(wzx, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(wzy, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(wzz, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(wzw, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(wwx, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(wwy, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(wwz, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(www, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xxxx, 0, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xxxy, 0, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xxxz, 0, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xxxw, 0, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xxyx, 0, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xxyy, 0, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xxyz, 0, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xxyw, 0, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xxzx, 0, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xxzy, 0, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xxzz, 0, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xxzw, 0, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xxwx, 0, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xxwy, 0, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xxwz, 0, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xxww, 0, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xyxx, 0, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xyxy, 0, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xyxz, 0, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xyxw, 0, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xyyx, 0, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xyyy, 0, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xyyz, 0, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xyyw, 0, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xyzx, 0, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xyzy, 0, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xyzz, 0, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xyzw, 0, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xywx, 0, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xywy, 0, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xywz, 0, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xyww, 0, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xzxx, 0, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xzxy, 0, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xzxz, 0, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xzxw, 0, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xzyx, 0, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xzyy, 0, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xzyz, 0, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xzyw, 0, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xzzx, 0, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xzzy, 0, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xzzz, 0, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xzzw, 0, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xzwx, 0, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xzwy, 0, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xzwz, 0, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xzww, 0, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xwxx, 0, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xwxy, 0, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xwxz, 0, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xwxw, 0, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xwyx, 0, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xwyy, 0, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xwyz, 0, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xwyw, 0, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xwzx, 0, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xwzy, 0, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xwzz, 0, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xwzw, 0, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(xwwx, 0, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(xwwy, 0, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(xwwz, 0, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(xwww, 0, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yxxx, 1, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yxxy, 1, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yxxz, 1, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yxxw, 1, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yxyx, 1, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yxyy, 1, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yxyz, 1, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yxyw, 1, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yxzx, 1, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yxzy, 1, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yxzz, 1, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yxzw, 1, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yxwx, 1, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yxwy, 1, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yxwz, 1, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yxww, 1, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yyxx, 1, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yyxy, 1, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yyxz, 1, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yyxw, 1, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yyyx, 1, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yyyy, 1, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yyyz, 1, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yyyw, 1, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yyzx, 1, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yyzy, 1, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yyzz, 1, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yyzw, 1, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yywx, 1, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yywy, 1, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yywz, 1, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yyww, 1, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yzxx, 1, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yzxy, 1, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yzxz, 1, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yzxw, 1, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yzyx, 1, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yzyy, 1, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yzyz, 1, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yzyw, 1, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yzzx, 1, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yzzy, 1, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yzzz, 1, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yzzw, 1, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(yzwx, 1, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(yzwy, 1, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(yzwz, 1, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(yzww, 1, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ywxx, 1, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ywxy, 1, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ywxz, 1, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ywxw, 1, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ywyx, 1, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ywyy, 1, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ywyz, 1, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ywyw, 1, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ywzx, 1, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ywzy, 1, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ywzz, 1, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ywzw, 1, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ywwx, 1, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ywwy, 1, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ywwz, 1, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ywww, 1, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zxxx, 2, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zxxy, 2, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zxxz, 2, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zxxw, 2, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zxyx, 2, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zxyy, 2, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zxyz, 2, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zxyw, 2, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zxzx, 2, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zxzy, 2, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zxzz, 2, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zxzw, 2, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zxwx, 2, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zxwy, 2, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zxwz, 2, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zxww, 2, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zyxx, 2, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zyxy, 2, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zyxz, 2, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zyxw, 2, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zyyx, 2, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zyyy, 2, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zyyz, 2, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zyyw, 2, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zyzx, 2, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zyzy, 2, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zyzz, 2, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zyzw, 2, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zywx, 2, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zywy, 2, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zywz, 2, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zyww, 2, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zzxx, 2, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zzxy, 2, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zzxz, 2, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zzxw, 2, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zzyx, 2, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zzyy, 2, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zzyz, 2, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zzyw, 2, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zzzx, 2, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zzzy, 2, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zzzz, 2, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zzzw, 2, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zzwx, 2, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zzwy, 2, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zzwz, 2, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zzww, 2, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zwxx, 2, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zwxy, 2, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zwxz, 2, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zwxw, 2, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zwyx, 2, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zwyy, 2, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zwyz, 2, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zwyw, 2, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zwzx, 2, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zwzy, 2, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zwzz, 2, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zwzw, 2, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(zwwx, 2, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(zwwy, 2, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(zwwz, 2, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(zwww, 2, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wxxx, 3, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wxxy, 3, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wxxz, 3, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wxxw, 3, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wxyx, 3, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wxyy, 3, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wxyz, 3, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wxyw, 3, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wxzx, 3, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wxzy, 3, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wxzz, 3, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wxzw, 3, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wxwx, 3, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wxwy, 3, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wxwz, 3, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wxww, 3, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wyxx, 3, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wyxy, 3, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wyxz, 3, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wyxw, 3, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wyyx, 3, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wyyy, 3, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wyyz, 3, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wyyw, 3, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wyzx, 3, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wyzy, 3, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wyzz, 3, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wyzw, 3, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wywx, 3, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wywy, 3, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wywz, 3, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wyww, 3, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wzxx, 3, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wzxy, 3, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wzxz, 3, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wzxw, 3, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wzyx, 3, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wzyy, 3, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wzyz, 3, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wzyw, 3, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wzzx, 3, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wzzy, 3, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wzzz, 3, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wzzw, 3, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wzwx, 3, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wzwy, 3, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wzwz, 3, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wzww, 3, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wwxx, 3, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wwxy, 3, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wwxz, 3, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wwxw, 3, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wwyx, 3, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wwyy, 3, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wwyz, 3, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wwyw, 3, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wwzx, 3, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wwzy, 3, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wwzz, 3, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wwzw, 3, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(wwwx, 3, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(wwwy, 3, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(wwwz, 3, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(wwww, 3, 3, 3, 3)

    PARAS_VEC_NAMED_SWIZZLE2(rr, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE2(rg, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE2(rb, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE2(ra, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE2(gr, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE2(gg, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE2(gb, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE2(ga, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE2(br, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE2(bg, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE2(bb, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE2(ba, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE2(ar, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE2(ag, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE2(ab, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE2(aa, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(rrr, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(rrg, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(rrb, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(rra, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(rgr, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(rgg, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(rgb, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(rga, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(rbr, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(rbg, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(rbb, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(rba, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(rar, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(rag, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(rab, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(raa, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(grr, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(grg, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(grb, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(gra, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(ggr, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(ggg, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(ggb, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(gga, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(gbr, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(gbg, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(gbb, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(gba, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(gar, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(gag, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(gab, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(gaa, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(brr, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(brg, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(brb, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(bra, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(bgr, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(bgg, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(bgb, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(bga, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(bbr, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(bbg, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(bbb, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(bba, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(bar, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(bag, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(bab, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(baa, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3(arr, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3(arg, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3(arb, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3(ara, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3(agr, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3(agg, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3(agb, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3(aga, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3(abr, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3(abg, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3(abb, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3(aba, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3(aar, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3(aag, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3(aab, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3(aaa, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rrrr, 0, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rrrg, 0, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rrrb, 0, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rrra, 0, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rrgr, 0, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rrgg, 0, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rrgb, 0, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rrga, 0, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rrbr, 0, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rrbg, 0, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rrbb, 0, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rrba, 0, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rrar, 0, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rrag, 0, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rrab, 0, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rraa, 0, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rgrr, 0, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rgrg, 0, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rgrb, 0, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rgra, 0, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rggr, 0, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rggg, 0, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rggb, 0, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rgga, 0, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rgbr, 0, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rgbg, 0, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rgbb, 0, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rgba, 0, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rgar, 0, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rgag, 0, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rgab, 0, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rgaa, 0, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rbrr, 0, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rbrg, 0, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rbrb, 0, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rbra, 0, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rbgr, 0, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rbgg, 0, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rbgb, 0, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rbga, 0, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rbbr, 0, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rbbg, 0, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rbbb, 0, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rbba, 0, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rbar, 0, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rbag, 0, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rbab, 0, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rbaa, 0, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rarr, 0, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rarg, 0, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rarb, 0, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(rara, 0, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ragr, 0, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ragg, 0, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ragb, 0, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(raga, 0, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(rabr, 0, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(rabg, 0, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(rabb, 0, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(raba, 0, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(raar, 0, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(raag, 0, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(raab, 0, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(raaa, 0, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(grrr, 1, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(grrg, 1, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(grrb, 1, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(grra, 1, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(grgr, 1, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(grgg, 1, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(grgb, 1, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(grga, 1, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(grbr, 1, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(grbg, 1, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(grbb, 1, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(grba, 1, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(grar, 1, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(grag, 1, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(grab, 1, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(graa, 1, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ggrr, 1, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ggrg, 1, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ggrb, 1, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ggra, 1, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gggr, 1, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gggg, 1, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gggb, 1, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ggga, 1, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ggbr, 1, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ggbg, 1, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ggbb, 1, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ggba, 1, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(ggar, 1, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(ggag, 1, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(ggab, 1, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(ggaa, 1, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gbrr, 1, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gbrg, 1, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gbrb, 1, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gbra, 1, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gbgr, 1, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gbgg, 1, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gbgb, 1, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gbga, 1, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gbbr, 1, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gbbg, 1, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gbbb, 1, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gbba, 1, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gbar, 1, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gbag, 1, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gbab, 1, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gbaa, 1, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(garr, 1, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(garg, 1, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(garb, 1, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gara, 1, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gagr, 1, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gagg, 1, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gagb, 1, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gaga, 1, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gabr, 1, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gabg, 1, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gabb, 1, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gaba, 1, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(gaar, 1, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(gaag, 1, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(gaab, 1, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(gaaa, 1, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(brrr, 2, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(brrg, 2, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(brrb, 2, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(brra, 2, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(brgr, 2, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(brgg, 2, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(brgb, 2, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(brga, 2, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(brbr, 2, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(brbg, 2, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(brbb, 2, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(brba, 2, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(brar, 2, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(brag, 2, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(brab, 2, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(braa, 2, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bgrr, 2, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bgrg, 2, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bgrb, 2, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bgra, 2, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bggr, 2, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bggg, 2, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bggb, 2, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bgga, 2, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bgbr, 2, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bgbg, 2, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bgbb, 2, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bgba, 2, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bgar, 2, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bgag, 2, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bgab, 2, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bgaa, 2, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bbrr, 2, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bbrg, 2, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bbrb, 2, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bbra, 2, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bbgr, 2, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bbgg, 2, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bbgb, 2, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bbga, 2, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bbbr, 2, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bbbg, 2, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bbbb, 2, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bbba, 2, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bbar, 2, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bbag, 2, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bbab, 2, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bbaa, 2, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(barr, 2, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(barg, 2, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(barb, 2, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(bara, 2, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(bagr, 2, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(bagg, 2, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(bagb, 2, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(baga, 2, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(babr, 2, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(babg, 2, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(babb, 2, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(baba, 2, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(baar, 2, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(baag, 2, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(baab, 2, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(baaa, 2, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(arrr, 3, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(arrg, 3, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(arrb, 3, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(arra, 3, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(argr, 3, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(argg, 3, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(argb, 3, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(arga, 3, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(arbr, 3, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(arbg, 3, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(arbb, 3, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(arba, 3, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(arar, 3, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(arag, 3, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(arab, 3, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(araa, 3, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(agrr, 3, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(agrg, 3, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(agrb, 3, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(agra, 3, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(aggr, 3, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(aggg, 3, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(aggb, 3, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(agga, 3, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(agbr, 3, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(agbg, 3, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(agbb, 3, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(agba, 3, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(agar, 3, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(agag, 3, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(agab, 3, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(agaa, 3, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(abrr, 3, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(abrg, 3, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(abrb, 3, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(abra, 3, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(abgr, 3, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(abgg, 3, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(abgb, 3, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(abga, 3, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(abbr, 3, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(abbg, 3, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(abbb, 3, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(abba, 3, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(abar, 3, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(abag, 3, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(abab, 3, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(abaa, 3, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4(aarr, 3, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4(aarg, 3, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4(aarb, 3, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4(aara, 3, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4(aagr, 3, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4(aagg, 3, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4(aagb, 3, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4(aaga, 3, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4(aabr, 3, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4(aabg, 3, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4(aabb, 3, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4(aaba, 3, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4(aaar, 3, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4(aaag, 3, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4(aaab, 3, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4(aaaa, 3, 3, 3, 3)

#undef PARAS_VEC_NAMED_SWIZZLE2
#undef PARAS_VEC_NAMED_SWIZZLE3
#undef PARAS_VEC_NAMED_SWIZZLE4

    PARAS_KERNEL_HD
    vec<T, (N + 1) / 2> lo() const noexcept {
        static_assert(N >= 2, "lo() requires N >= 2");
        vec<T, (N + 1) / 2> result;
        for (int i = 0; i < (N + 1) / 2; ++i)
            result.data[i] = data[i];
        return result;
    }

    PARAS_KERNEL_HD
    vec<T, N / 2> hi() const noexcept {
        static_assert(N >= 2, "hi() requires N >= 2");
        vec<T, N / 2> result;
        for (int i = 0; i < N / 2; ++i)
            result.data[i] = data[(N + 1) / 2 + i];
        return result;
    }

    PARAS_KERNEL_HD
    vec<T, N / 2> odd() const noexcept {
        static_assert(N >= 2, "odd() requires N >= 2");
        vec<T, N / 2> result;
        for (int i = 0; i < N / 2; ++i)
            result.data[i] = data[2 * i + 1];
        return result;
    }

    PARAS_KERNEL_HD
    vec<T, (N + 1) / 2> even() const noexcept {
        static_assert(N >= 2, "even() requires N >= 2");
        vec<T, (N + 1) / 2> result;
        for (int i = 0; i < (N + 1) / 2; ++i)
            result.data[i] = data[2 * i];
        return result;
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

    template <auto... Indices>
    PARAS_KERNEL_HD vec<T, static_cast<int>(sizeof...(Indices))> swizzle() const noexcept {
        return vec<T, static_cast<int>(sizeof...(Indices))>{data[static_cast<int>(Indices)]...};
    }

    template <elem... Indices>
    struct swizzle_ref {
        T* base;

        using value_type = T;
        using element_type = T;

        template <typename U, typename = void>
        struct paras_swizzle_op_has_size : std::false_type {};
        template <typename U>
        struct paras_swizzle_op_has_size<U, std::void_t<decltype(U::size())>> : std::true_type {};
        template <typename U>
        static constexpr bool paras_swizzle_op_has_size_v = paras_swizzle_op_has_size<U>::value;

#define PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP(OP)                                            \
    template <typename RhsT, typename = std::enable_if_t<std::is_convertible_v<RhsT, T>>>          \
    PARAS_KERNEL_HD auto operator OP(const RhsT& rhs) const noexcept {                             \
        using SelfVec = vec<T, static_cast<int>(sizeof...(Indices))>;                              \
        return static_cast<SelfVec>(*this) OP static_cast<T>(rhs);                                 \
    }                                                                                              \
    template <typename RhsT,                                                                       \
              typename = std::enable_if_t<                                                         \
                  !std::is_convertible_v<RhsT, T> && static_cast<int>(sizeof...(Indices)) == 1 &&  \
                  paras_swizzle_op_has_size_v<RhsT> && (static_cast<int>(RhsT::size()) != 1)>,     \
              typename = void, typename = void>                                                    \
    PARAS_KERNEL_HD auto operator OP(const RhsT& rhs) const noexcept {                             \
        return static_cast<T>(*this) OP rhs;                                                       \
    }                                                                                              \
    template <typename RhsT,                                                                       \
              typename = std::enable_if_t<!std::is_convertible_v<RhsT, T> &&                       \
                                          !((static_cast<int>(sizeof...(Indices)) == 1) &&         \
                                            paras_swizzle_op_has_size_v<RhsT> &&                   \
                                            (static_cast<int>(RhsT::size()) != 1))>,               \
              typename = void>                                                                     \
    PARAS_KERNEL_HD auto operator OP(const RhsT& rhs) const noexcept {                             \
        using SelfVec = vec<T, static_cast<int>(sizeof...(Indices))>;                              \
        return static_cast<SelfVec>(*this) OP static_cast<SelfVec>(rhs);                           \
    }

        PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP(==)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP(!=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP(<)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP(>)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP(<=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP(>=)

#undef PARAS_SYCL_DEFINE_SWIZZLE_REF_RELATIONAL_OP

        template <typename RhsT, typename = std::enable_if_t<std::is_convertible_v<RhsT, T>>>
        PARAS_KERNEL_HD auto operator&&(const RhsT& rhs) const noexcept {
            using SelfVec = vec<T, static_cast<int>(sizeof...(Indices))>;
            return static_cast<SelfVec>(*this) && static_cast<T>(rhs);
        }

        template <
            typename RhsT,
            typename = std::enable_if_t<
                !std::is_convertible_v<RhsT, T> && static_cast<int>(sizeof...(Indices)) == 1 &&
                paras_swizzle_op_has_size_v<RhsT> && (static_cast<int>(RhsT::size()) != 1)>,
            typename = void, typename = void>
        PARAS_KERNEL_HD auto operator&&(const RhsT& rhs) const noexcept {
            return static_cast<T>(*this) && rhs;
        }

        template <typename RhsT,
                  typename = std::enable_if_t<!std::is_convertible_v<RhsT, T> &&
                                              !((static_cast<int>(sizeof...(Indices)) == 1) &&
                                                paras_swizzle_op_has_size_v<RhsT> &&
                                                (static_cast<int>(RhsT::size()) != 1))>,
                  typename = void>
        PARAS_KERNEL_HD auto operator&&(const RhsT& rhs) const noexcept {
            using SelfVec = vec<T, static_cast<int>(sizeof...(Indices))>;
            return static_cast<SelfVec>(*this) && static_cast<SelfVec>(rhs);
        }

        template <typename RhsT, typename = std::enable_if_t<std::is_convertible_v<RhsT, T>>>
        PARAS_KERNEL_HD auto operator||(const RhsT& rhs) const noexcept {
            using SelfVec = vec<T, static_cast<int>(sizeof...(Indices))>;
            return static_cast<SelfVec>(*this) || static_cast<T>(rhs);
        }

        template <
            typename RhsT,
            typename = std::enable_if_t<
                !std::is_convertible_v<RhsT, T> && static_cast<int>(sizeof...(Indices)) == 1 &&
                paras_swizzle_op_has_size_v<RhsT> && (static_cast<int>(RhsT::size()) != 1)>,
            typename = void, typename = void>
        PARAS_KERNEL_HD auto operator||(const RhsT& rhs) const noexcept {
            return static_cast<T>(*this) || rhs;
        }

        template <typename RhsT,
                  typename = std::enable_if_t<!std::is_convertible_v<RhsT, T> &&
                                              !((static_cast<int>(sizeof...(Indices)) == 1) &&
                                                paras_swizzle_op_has_size_v<RhsT> &&
                                                (static_cast<int>(RhsT::size()) != 1))>,
                  typename = void>
        PARAS_KERNEL_HD auto operator||(const RhsT& rhs) const noexcept {
            using SelfVec = vec<T, static_cast<int>(sizeof...(Indices))>;
            return static_cast<SelfVec>(*this) || static_cast<SelfVec>(rhs);
        }

#define PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(OP)                                     \
    PARAS_KERNEL_HD                                                                                \
    auto operator OP(const T& rhs) const noexcept {                                                \
        return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this) OP rhs;            \
    }

        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(+)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(-)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(*)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(/)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(>>)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(<<)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(^)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(|)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(&)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP(%)

#undef PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SCALAR_OP

#define PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(OP)                                        \
    PARAS_KERNEL_HD                                                                                \
    auto operator OP(const vec<T, static_cast<int>(sizeof...(Indices))>& rhs) const noexcept {     \
        return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this) OP rhs;            \
    }

        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(+)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(-)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(*)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(/)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(&)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(|)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(^)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(<<)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(>>)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP(%)
#undef PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_VEC_OP

#define PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(OP)                                    \
    template <typename RhsSwizzleRefT,                                                             \
              typename = std::enable_if_t<                                                         \
                  paras_is_swizzle_ref<RhsSwizzleRefT>::value&& static_cast<int>(                  \
                      RhsSwizzleRefT::size()) == static_cast<int>(sizeof...(Indices))>>            \
    PARAS_KERNEL_HD auto operator OP(const RhsSwizzleRefT& rhs) const noexcept {                   \
        return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this)                    \
            OP static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(rhs);                     \
    }

        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(+)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(-)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(*)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(/)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(^)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(|)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(<<)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(>>)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(&)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP(%)
#undef PARAS_SYCL_DEFINE_SWIZZLE_REF_ARITHMETIC_SWIZZLE_OP

        PARAS_KERNEL_HD
        swizzle_ref& operator=(const T& scalar) noexcept {
            ((base[static_cast<int>(Indices)] = scalar), ...);
            return *this;
        }

        PARAS_KERNEL_HD
        swizzle_ref& operator=(const vec<T, static_cast<int>(sizeof...(Indices))>& rhs) noexcept {
            int i = 0;
            ((base[static_cast<int>(Indices)] = rhs[i++]), ...);
            return *this;
        }

        PARAS_KERNEL_HD
        swizzle_ref& operator=(const swizzle_ref& rhs) noexcept {
            return (*this) = static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(rhs);
        }

        PARAS_KERNEL_HD
        T& operator[](int i) noexcept {
            static constexpr int idx[] = {static_cast<int>(Indices)...};
            return base[idx[i]];
        }
        PARAS_KERNEL_HD
        const T& operator[](int i) const noexcept {
            static constexpr int idx[] = {static_cast<int>(Indices)...};
            return base[idx[i]];
        }

        PARAS_KERNEL_HD
        vec<T, static_cast<int>(sizeof...(Indices))> operator+() const noexcept {
            return +static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this);
        }
        PARAS_KERNEL_HD
        vec<T, static_cast<int>(sizeof...(Indices))> operator-() const noexcept {
            return -static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this);
        }
        PARAS_KERNEL_HD
        vec<T, static_cast<int>(sizeof...(Indices))> operator~() const noexcept {
            return ~static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this);
        }

        PARAS_KERNEL_HD
        auto operator!() const noexcept {
            return !static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this);
        }

        template <typename ConvertT, rounding_mode Mode = rounding_mode::automatic>
        PARAS_KERNEL_HD vec<ConvertT, static_cast<int>(sizeof...(Indices))>
        convert() const noexcept {
            return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this)
                .template convert<ConvertT, Mode>();
        }

        template <typename asVecType>
        PARAS_KERNEL_HD asVecType as() const noexcept {
            return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this)
                .template as<asVecType>();
        }

        PARAS_KERNEL_HD
        swizzle_ref& operator++() noexcept {
            ((++base[static_cast<int>(Indices)]), ...);
            return *this;
        }
        PARAS_KERNEL_HD
        vec<T, static_cast<int>(sizeof...(Indices))> operator++(int) noexcept {
            auto tmp = static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this);
            ++(*this);
            return tmp;
        }
        PARAS_KERNEL_HD
        swizzle_ref& operator--() noexcept {
            ((--base[static_cast<int>(Indices)]), ...);
            return *this;
        }
        PARAS_KERNEL_HD
        vec<T, static_cast<int>(sizeof...(Indices))> operator--(int) noexcept {
            auto tmp = static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this);
            --(*this);
            return tmp;
        }

        template <int M = static_cast<int>(sizeof...(Indices)), std::enable_if_t<(M > 1), int> = 0>
        PARAS_KERNEL_HD operator vec<T, M>() const noexcept {
            return vec<T, M>{base[static_cast<int>(Indices)]...};
        }

        template <int M = static_cast<int>(sizeof...(Indices)), std::enable_if_t<M == 1, int> = 0>
        PARAS_KERNEL_HD explicit operator vec<T, M>() const noexcept {
            return vec<T, M>{base[static_cast<int>(Indices)]...};
        }

        template <int M = static_cast<int>(sizeof...(Indices)), typename = std::enable_if_t<M == 1>>
        PARAS_KERNEL_HD operator T() const noexcept {
            T result;
            ((result = base[static_cast<int>(Indices)]), ...);
            return result;
        }

#define PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(OP)                                              \
    PARAS_KERNEL_HD                                                                                \
    swizzle_ref& operator OP(const T& rhs) noexcept {                                              \
        ((base[static_cast<int>(Indices)] OP rhs), ...);                                           \
        return *this;                                                                              \
    }                                                                                              \
    PARAS_KERNEL_HD                                                                                \
    swizzle_ref& operator OP(const vec<T, static_cast<int>(sizeof...(Indices))>& rhs) noexcept {   \
        int paras_i = 0;                                                                           \
        ((base[static_cast<int>(Indices)] OP rhs[paras_i++]), ...);                                \
        return *this;                                                                              \
    }                                                                                              \
    PARAS_KERNEL_HD                                                                                \
    swizzle_ref& operator OP(const swizzle_ref& rhs) noexcept {                                    \
        return (*this)OP static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(rhs);           \
    }

        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(+=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(-=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(*=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(/=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(%=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(&=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(|=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(^=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(<<=)
        PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP(>>=)

#undef PARAS_SYCL_DEFINE_SWIZZLE_REF_COMPOUND_OP

        PARAS_KERNEL_HD
        static constexpr std::size_t size() noexcept {
            return static_cast<std::size_t>(sizeof...(Indices));
        }

        PARAS_KERNEL_HD
        std::size_t byte_size() const noexcept {
            return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this).byte_size();
        }

        PARAS_KERNEL_HD
        vec<T, (static_cast<int>(sizeof...(Indices)) + 1) / 2> lo() const noexcept {
            return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this).lo();
        }

        PARAS_KERNEL_HD
        vec<T, static_cast<int>(sizeof...(Indices)) / 2> hi() const noexcept {
            return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this).hi();
        }

        PARAS_KERNEL_HD
        vec<T, static_cast<int>(sizeof...(Indices)) / 2> odd() const noexcept {
            return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this).odd();
        }

        PARAS_KERNEL_HD
        vec<T, (static_cast<int>(sizeof...(Indices)) + 1) / 2> even() const noexcept {
            return static_cast<vec<T, static_cast<int>(sizeof...(Indices))>>(*this).even();
        }
    };

    template <auto... Indices>
    PARAS_KERNEL_HD swizzle_ref<static_cast<elem>(Indices)...> swizzle() noexcept {
        return swizzle_ref<static_cast<elem>(Indices)...>{data};
    }

#define PARAS_VEC_NAMED_SWIZZLE2_MUT(NAME, I0, I1)                                                 \
    PARAS_KERNEL_HD                                                                                \
    swizzle_ref<static_cast<elem>(I0), static_cast<elem>(I1)> NAME() noexcept {                    \
        return swizzle_ref<static_cast<elem>(I0), static_cast<elem>(I1)>{data};                    \
    }

#define PARAS_VEC_NAMED_SWIZZLE3_MUT(NAME, I0, I1, I2)                                             \
    PARAS_KERNEL_HD                                                                                \
    swizzle_ref<static_cast<elem>(I0), static_cast<elem>(I1), static_cast<elem>(I2)>               \
    NAME() noexcept {                                                                              \
        return swizzle_ref<static_cast<elem>(I0), static_cast<elem>(I1), static_cast<elem>(I2)>{   \
            data};                                                                                 \
    }

#define PARAS_VEC_NAMED_SWIZZLE4_MUT(NAME, I0, I1, I2, I3)                                         \
    PARAS_KERNEL_HD                                                                                \
    swizzle_ref<static_cast<elem>(I0), static_cast<elem>(I1), static_cast<elem>(I2),               \
                static_cast<elem>(I3)>                                                             \
    NAME() noexcept {                                                                              \
        return swizzle_ref<static_cast<elem>(I0), static_cast<elem>(I1), static_cast<elem>(I2),    \
                           static_cast<elem>(I3)>{data};                                           \
    }

    PARAS_VEC_NAMED_SWIZZLE2_MUT(xx, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(xy, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(xz, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(xw, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(yx, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(yy, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(yz, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(yw, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(zx, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(zy, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(zz, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(zw, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(wx, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(wy, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(wz, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(ww, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xxx, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xxy, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xxz, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xxw, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xyx, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xyy, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xyz, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xyw, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xzx, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xzy, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xzz, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xzw, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xwx, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xwy, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xwz, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(xww, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yxx, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yxy, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yxz, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yxw, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yyx, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yyy, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yyz, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yyw, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yzx, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yzy, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yzz, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yzw, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(ywx, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(ywy, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(ywz, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(yww, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zxx, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zxy, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zxz, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zxw, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zyx, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zyy, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zyz, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zyw, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zzx, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zzy, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zzz, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zzw, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zwx, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zwy, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zwz, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(zww, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wxx, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wxy, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wxz, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wxw, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wyx, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wyy, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wyz, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wyw, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wzx, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wzy, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wzz, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wzw, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wwx, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wwy, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(wwz, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(www, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxxx, 0, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxxy, 0, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxxz, 0, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxxw, 0, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxyx, 0, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxyy, 0, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxyz, 0, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxyw, 0, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxzx, 0, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxzy, 0, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxzz, 0, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxzw, 0, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxwx, 0, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxwy, 0, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxwz, 0, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xxww, 0, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyxx, 0, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyxy, 0, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyxz, 0, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyxw, 0, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyyx, 0, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyyy, 0, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyyz, 0, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyyw, 0, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyzx, 0, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyzy, 0, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyzz, 0, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyzw, 0, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xywx, 0, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xywy, 0, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xywz, 0, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xyww, 0, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzxx, 0, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzxy, 0, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzxz, 0, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzxw, 0, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzyx, 0, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzyy, 0, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzyz, 0, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzyw, 0, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzzx, 0, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzzy, 0, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzzz, 0, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzzw, 0, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzwx, 0, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzwy, 0, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzwz, 0, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xzww, 0, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwxx, 0, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwxy, 0, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwxz, 0, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwxw, 0, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwyx, 0, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwyy, 0, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwyz, 0, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwyw, 0, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwzx, 0, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwzy, 0, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwzz, 0, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwzw, 0, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwwx, 0, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwwy, 0, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwwz, 0, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(xwww, 0, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxxx, 1, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxxy, 1, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxxz, 1, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxxw, 1, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxyx, 1, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxyy, 1, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxyz, 1, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxyw, 1, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxzx, 1, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxzy, 1, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxzz, 1, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxzw, 1, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxwx, 1, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxwy, 1, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxwz, 1, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yxww, 1, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyxx, 1, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyxy, 1, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyxz, 1, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyxw, 1, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyyx, 1, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyyy, 1, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyyz, 1, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyyw, 1, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyzx, 1, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyzy, 1, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyzz, 1, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyzw, 1, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yywx, 1, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yywy, 1, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yywz, 1, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yyww, 1, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzxx, 1, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzxy, 1, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzxz, 1, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzxw, 1, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzyx, 1, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzyy, 1, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzyz, 1, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzyw, 1, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzzx, 1, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzzy, 1, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzzz, 1, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzzw, 1, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzwx, 1, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzwy, 1, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzwz, 1, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(yzww, 1, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywxx, 1, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywxy, 1, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywxz, 1, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywxw, 1, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywyx, 1, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywyy, 1, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywyz, 1, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywyw, 1, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywzx, 1, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywzy, 1, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywzz, 1, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywzw, 1, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywwx, 1, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywwy, 1, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywwz, 1, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ywww, 1, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxxx, 2, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxxy, 2, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxxz, 2, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxxw, 2, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxyx, 2, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxyy, 2, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxyz, 2, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxyw, 2, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxzx, 2, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxzy, 2, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxzz, 2, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxzw, 2, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxwx, 2, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxwy, 2, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxwz, 2, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zxww, 2, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyxx, 2, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyxy, 2, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyxz, 2, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyxw, 2, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyyx, 2, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyyy, 2, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyyz, 2, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyyw, 2, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyzx, 2, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyzy, 2, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyzz, 2, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyzw, 2, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zywx, 2, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zywy, 2, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zywz, 2, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zyww, 2, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzxx, 2, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzxy, 2, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzxz, 2, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzxw, 2, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzyx, 2, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzyy, 2, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzyz, 2, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzyw, 2, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzzx, 2, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzzy, 2, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzzz, 2, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzzw, 2, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzwx, 2, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzwy, 2, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzwz, 2, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zzww, 2, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwxx, 2, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwxy, 2, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwxz, 2, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwxw, 2, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwyx, 2, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwyy, 2, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwyz, 2, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwyw, 2, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwzx, 2, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwzy, 2, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwzz, 2, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwzw, 2, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwwx, 2, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwwy, 2, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwwz, 2, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(zwww, 2, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxxx, 3, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxxy, 3, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxxz, 3, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxxw, 3, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxyx, 3, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxyy, 3, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxyz, 3, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxyw, 3, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxzx, 3, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxzy, 3, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxzz, 3, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxzw, 3, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxwx, 3, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxwy, 3, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxwz, 3, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wxww, 3, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyxx, 3, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyxy, 3, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyxz, 3, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyxw, 3, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyyx, 3, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyyy, 3, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyyz, 3, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyyw, 3, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyzx, 3, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyzy, 3, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyzz, 3, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyzw, 3, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wywx, 3, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wywy, 3, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wywz, 3, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wyww, 3, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzxx, 3, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzxy, 3, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzxz, 3, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzxw, 3, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzyx, 3, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzyy, 3, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzyz, 3, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzyw, 3, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzzx, 3, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzzy, 3, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzzz, 3, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzzw, 3, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzwx, 3, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzwy, 3, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzwz, 3, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wzww, 3, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwxx, 3, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwxy, 3, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwxz, 3, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwxw, 3, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwyx, 3, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwyy, 3, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwyz, 3, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwyw, 3, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwzx, 3, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwzy, 3, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwzz, 3, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwzw, 3, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwwx, 3, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwwy, 3, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwwz, 3, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(wwww, 3, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(rr, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(rg, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(rb, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(ra, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(gr, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(gg, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(gb, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(ga, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(br, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(bg, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(bb, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(ba, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(ar, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(ag, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(ab, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE2_MUT(aa, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rrr, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rrg, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rrb, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rra, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rgr, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rgg, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rgb, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rga, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rbr, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rbg, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rbb, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rba, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rar, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rag, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(rab, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(raa, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(grr, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(grg, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(grb, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gra, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(ggr, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(ggg, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(ggb, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gga, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gbr, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gbg, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gbb, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gba, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gar, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gag, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gab, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(gaa, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(brr, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(brg, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(brb, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bra, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bgr, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bgg, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bgb, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bga, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bbr, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bbg, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bbb, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bba, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bar, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bag, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(bab, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(baa, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(arr, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(arg, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(arb, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(ara, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(agr, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(agg, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(agb, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(aga, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(abr, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(abg, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(abb, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(aba, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(aar, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(aag, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(aab, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE3_MUT(aaa, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrrr, 0, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrrg, 0, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrrb, 0, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrra, 0, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrgr, 0, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrgg, 0, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrgb, 0, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrga, 0, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrbr, 0, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrbg, 0, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrbb, 0, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrba, 0, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrar, 0, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrag, 0, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rrab, 0, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rraa, 0, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgrr, 0, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgrg, 0, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgrb, 0, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgra, 0, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rggr, 0, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rggg, 0, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rggb, 0, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgga, 0, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgbr, 0, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgbg, 0, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgbb, 0, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgba, 0, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgar, 0, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgag, 0, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgab, 0, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rgaa, 0, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbrr, 0, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbrg, 0, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbrb, 0, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbra, 0, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbgr, 0, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbgg, 0, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbgb, 0, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbga, 0, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbbr, 0, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbbg, 0, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbbb, 0, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbba, 0, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbar, 0, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbag, 0, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbab, 0, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rbaa, 0, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rarr, 0, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rarg, 0, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rarb, 0, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rara, 0, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ragr, 0, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ragg, 0, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ragb, 0, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(raga, 0, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rabr, 0, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rabg, 0, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(rabb, 0, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(raba, 0, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(raar, 0, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(raag, 0, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(raab, 0, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(raaa, 0, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grrr, 1, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grrg, 1, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grrb, 1, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grra, 1, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grgr, 1, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grgg, 1, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grgb, 1, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grga, 1, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grbr, 1, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grbg, 1, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grbb, 1, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grba, 1, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grar, 1, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grag, 1, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(grab, 1, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(graa, 1, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggrr, 1, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggrg, 1, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggrb, 1, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggra, 1, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gggr, 1, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gggg, 1, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gggb, 1, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggga, 1, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggbr, 1, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggbg, 1, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggbb, 1, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggba, 1, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggar, 1, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggag, 1, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggab, 1, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(ggaa, 1, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbrr, 1, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbrg, 1, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbrb, 1, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbra, 1, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbgr, 1, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbgg, 1, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbgb, 1, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbga, 1, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbbr, 1, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbbg, 1, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbbb, 1, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbba, 1, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbar, 1, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbag, 1, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbab, 1, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gbaa, 1, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(garr, 1, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(garg, 1, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(garb, 1, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gara, 1, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gagr, 1, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gagg, 1, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gagb, 1, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gaga, 1, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gabr, 1, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gabg, 1, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gabb, 1, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gaba, 1, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gaar, 1, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gaag, 1, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gaab, 1, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(gaaa, 1, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brrr, 2, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brrg, 2, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brrb, 2, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brra, 2, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brgr, 2, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brgg, 2, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brgb, 2, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brga, 2, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brbr, 2, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brbg, 2, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brbb, 2, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brba, 2, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brar, 2, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brag, 2, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(brab, 2, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(braa, 2, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgrr, 2, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgrg, 2, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgrb, 2, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgra, 2, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bggr, 2, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bggg, 2, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bggb, 2, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgga, 2, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgbr, 2, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgbg, 2, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgbb, 2, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgba, 2, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgar, 2, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgag, 2, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgab, 2, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bgaa, 2, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbrr, 2, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbrg, 2, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbrb, 2, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbra, 2, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbgr, 2, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbgg, 2, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbgb, 2, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbga, 2, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbbr, 2, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbbg, 2, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbbb, 2, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbba, 2, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbar, 2, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbag, 2, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbab, 2, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bbaa, 2, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(barr, 2, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(barg, 2, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(barb, 2, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bara, 2, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bagr, 2, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bagg, 2, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(bagb, 2, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(baga, 2, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(babr, 2, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(babg, 2, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(babb, 2, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(baba, 2, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(baar, 2, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(baag, 2, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(baab, 2, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(baaa, 2, 3, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arrr, 3, 0, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arrg, 3, 0, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arrb, 3, 0, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arra, 3, 0, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(argr, 3, 0, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(argg, 3, 0, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(argb, 3, 0, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arga, 3, 0, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arbr, 3, 0, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arbg, 3, 0, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arbb, 3, 0, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arba, 3, 0, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arar, 3, 0, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arag, 3, 0, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(arab, 3, 0, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(araa, 3, 0, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agrr, 3, 1, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agrg, 3, 1, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agrb, 3, 1, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agra, 3, 1, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aggr, 3, 1, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aggg, 3, 1, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aggb, 3, 1, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agga, 3, 1, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agbr, 3, 1, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agbg, 3, 1, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agbb, 3, 1, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agba, 3, 1, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agar, 3, 1, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agag, 3, 1, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agab, 3, 1, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(agaa, 3, 1, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abrr, 3, 2, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abrg, 3, 2, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abrb, 3, 2, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abra, 3, 2, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abgr, 3, 2, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abgg, 3, 2, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abgb, 3, 2, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abga, 3, 2, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abbr, 3, 2, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abbg, 3, 2, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abbb, 3, 2, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abba, 3, 2, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abar, 3, 2, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abag, 3, 2, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abab, 3, 2, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(abaa, 3, 2, 3, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aarr, 3, 3, 0, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aarg, 3, 3, 0, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aarb, 3, 3, 0, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aara, 3, 3, 0, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aagr, 3, 3, 1, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aagg, 3, 3, 1, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aagb, 3, 3, 1, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aaga, 3, 3, 1, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aabr, 3, 3, 2, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aabg, 3, 3, 2, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aabb, 3, 3, 2, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aaba, 3, 3, 2, 3)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aaar, 3, 3, 3, 0)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aaag, 3, 3, 3, 1)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aaab, 3, 3, 3, 2)
    PARAS_VEC_NAMED_SWIZZLE4_MUT(aaaa, 3, 3, 3, 3)

#undef PARAS_VEC_NAMED_SWIZZLE2_MUT
#undef PARAS_VEC_NAMED_SWIZZLE3_MUT
#undef PARAS_VEC_NAMED_SWIZZLE4_MUT

    template <typename ConvertT, rounding_mode Mode = rounding_mode::automatic>
    PARAS_KERNEL_HD vec<ConvertT, N> convert() const noexcept {
        vec<ConvertT, N> result;
        for (int i = 0; i < N; ++i)
            result.data[i] = static_cast<ConvertT>(data[i]);
        return result;
    }

    template <typename asVecType>
    PARAS_KERNEL_HD asVecType as() const noexcept {
        static_assert(
            sizeof(asVecType) == sizeof(vec),
            "as<>() requires the source and destination vec to have the same total byte size");
        asVecType result;
        std::memcpy(&result, this, sizeof(vec));
        return result;
    }

    template <int M = N, typename = std::enable_if_t<M == 1>>
    PARAS_KERNEL_HD operator T() const noexcept {
        return data[0];
    }
};
template <typename T, typename... U>
vec(T, U...) -> vec<T, sizeof...(U) + 1>;

template <typename T, int N>
PARAS_KERNEL_HD constexpr T dot(const vec<T, N>& a, const vec<T, N>& b) noexcept {
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

class half {
public:
    PARAS_KERNEL_HD
    constexpr half() noexcept : bits_(0) {}

    PARAS_KERNEL_HD
    half(float v) noexcept : bits_(float_to_half_bits(v)) {}

    PARAS_KERNEL_HD
    operator float() const noexcept { return half_bits_to_float(bits_); }

    PARAS_KERNEL_HD
    half& operator=(float v) noexcept {
        bits_ = float_to_half_bits(v);
        return *this;
    }

    PARAS_KERNEL_HD
    half& operator++() noexcept {
        *this = static_cast<float>(*this) + 1.0f;
        return *this;
    }
    PARAS_KERNEL_HD
    half operator++(int) noexcept {
        half tmp(*this);
        ++(*this);
        return tmp;
    }

    PARAS_KERNEL_HD
    half& operator--() noexcept {
        *this = static_cast<float>(*this) - 1.0f;
        return *this;
    }
    PARAS_KERNEL_HD
    half operator--(int) noexcept {
        half tmp(*this);
        --(*this);
        return tmp;
    }

    PARAS_KERNEL_HD
    half& operator+=(half rhs) noexcept {
        *this = static_cast<float>(*this) + static_cast<float>(rhs);
        return *this;
    }
    PARAS_KERNEL_HD
    half& operator-=(half rhs) noexcept {
        *this = static_cast<float>(*this) - static_cast<float>(rhs);
        return *this;
    }

    PARAS_KERNEL_HD
    half& operator*=(half rhs) noexcept {
        *this = static_cast<float>(*this) * static_cast<float>(rhs);
        return *this;
    }
    PARAS_KERNEL_HD
    half& operator/=(half rhs) noexcept {
        *this = static_cast<float>(*this) / static_cast<float>(rhs);
        return *this;
    }

private:
    std::uint16_t bits_;

    PARAS_KERNEL_HD
    static std::uint16_t float_to_half_bits(float f) noexcept {
        std::uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        std::uint32_t sign = (x >> 16) & 0x8000u;
        std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
        std::uint32_t mant = x & 0x7FFFFFu;

        if (((x >> 23) & 0xFF) == 0xFF) {
            return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
        }
        if (exp <= 0) {
            if (exp < -10)
                return static_cast<std::uint16_t>(sign);
            mant |= 0x800000u;
            std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
            std::uint32_t half_mant = mant >> shift;
            if ((mant >> (shift - 1)) & 1u)
                ++half_mant;
            return static_cast<std::uint16_t>(sign | half_mant);
        }
        if (exp >= 0x1F) {
            return static_cast<std::uint16_t>(sign | 0x7C00u);
        }
        std::uint32_t half_mant = mant >> 13;
        if (mant & 0x1000u) {
            ++half_mant;
            if (half_mant == 0x400u) {
                half_mant = 0;
                ++exp;
                if (exp >= 0x1F)
                    return static_cast<std::uint16_t>(sign | 0x7C00u);
            }
        }
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                          half_mant);
    }

    PARAS_KERNEL_HD
    static float half_bits_to_float(std::uint16_t h) noexcept {
        std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
        std::uint32_t exp = (h >> 10) & 0x1Fu;
        std::uint32_t mant = h & 0x3FFu;
        std::uint32_t bits;

        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                std::uint32_t e = 1;
                while ((mant & 0x400u) == 0u) {
                    mant <<= 1;
                    --e;
                }
                mant &= 0x3FFu;
                std::uint32_t fexp = static_cast<std::uint32_t>(127 - 15 + static_cast<int>(e));
                bits = sign | (fexp << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            bits = sign | 0x7F800000u | (mant << 13);
        } else {
            std::uint32_t fexp = exp - 15 + 127;
            bits = sign | (fexp << 23) | (mant << 13);
        }

        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

using char2 = vec<std::int8_t, 2>;
using char3 = vec<std::int8_t, 3>;
using char4 = vec<std::int8_t, 4>;
using char8 = vec<std::int8_t, 8>;
using char16 = vec<std::int8_t, 16>;

using uchar2 = vec<std::uint8_t, 2>;
using uchar3 = vec<std::uint8_t, 3>;
using uchar4 = vec<std::uint8_t, 4>;
using uchar8 = vec<std::uint8_t, 8>;
using uchar16 = vec<std::uint8_t, 16>;

using short2 = vec<std::int16_t, 2>;
using short3 = vec<std::int16_t, 3>;
using short4 = vec<std::int16_t, 4>;
using short8 = vec<std::int16_t, 8>;
using short16 = vec<std::int16_t, 16>;

using ushort2 = vec<std::uint16_t, 2>;
using ushort3 = vec<std::uint16_t, 3>;
using ushort4 = vec<std::uint16_t, 4>;
using ushort8 = vec<std::uint16_t, 8>;
using ushort16 = vec<std::uint16_t, 16>;

using int8 = vec<std::int32_t, 8>;
using int16 = vec<std::int32_t, 16>;
using uint8 = vec<std::uint32_t, 8>;
using uint16 = vec<std::uint32_t, 16>;

using long2 = vec<std::int64_t, 2>;
using long3 = vec<std::int64_t, 3>;
using long4 = vec<std::int64_t, 4>;
using long8 = vec<std::int64_t, 8>;
using long16 = vec<std::int64_t, 16>;

using ulong2 = vec<std::uint64_t, 2>;
using ulong3 = vec<std::uint64_t, 3>;
using ulong4 = vec<std::uint64_t, 4>;
using ulong8 = vec<std::uint64_t, 8>;
using ulong16 = vec<std::uint64_t, 16>;

using float8 = vec<float, 8>;
using float16 = vec<float, 16>;

using double2 = vec<double, 2>;
using double3 = vec<double, 3>;
using double4 = vec<double, 4>;
using double8 = vec<double, 8>;
using double16 = vec<double, 16>;

template <typename T, std::size_t N>
class marray {
public:
    using value_type = T;

    T data[N];

    PARAS_KERNEL_HD
    constexpr marray() noexcept : data{} {}

    PARAS_KERNEL_HD
    explicit constexpr marray(const T& v) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] = v;
    }

    template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == N>>
    PARAS_KERNEL_HD constexpr marray(Args... args) noexcept : data{static_cast<T>(args)...} {}

    PARAS_KERNEL_HD
    static constexpr std::size_t size() noexcept { return N; }

    PARAS_KERNEL_HD
    T& operator[](std::size_t i) noexcept { return data[i]; }

    PARAS_KERNEL_HD
    const T& operator[](std::size_t i) const noexcept { return data[i]; }

    using iterator = T*;
    using const_iterator = const T*;

    PARAS_KERNEL_HD
    iterator begin() noexcept { return data; }
    PARAS_KERNEL_HD
    const_iterator begin() const noexcept { return data; }

    PARAS_KERNEL_HD
    iterator end() noexcept { return data + N; }
    PARAS_KERNEL_HD
    const_iterator end() const noexcept { return data + N; }

    PARAS_KERNEL_HD
    marray& operator+=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] += rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator+=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] += rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator-=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] -= rhs.data[i];
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator-=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] -= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator*=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] *= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator*=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] *= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator/=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] /= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator/=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] /= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator%=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] %= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator%=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] %= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator&=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] &= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator&=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] &= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator|=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] |= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator|=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] |= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator^=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] ^= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator^=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] ^= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator<<=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] <<= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator<<=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] <<= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator>>=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] >>= rhs.data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray& operator>>=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] >>= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray operator+() const noexcept { return *this; }
    PARAS_KERNEL_HD
    marray operator-() const noexcept {
        marray result;
        for (std::size_t i = 0; i < N; ++i)
            result.data[i] = -data[i];
        return result;
    }

    PARAS_KERNEL_HD
    marray<bool, N> operator!() const noexcept {
        marray<bool, N> result{};
        for (std::size_t i = 0; i < N; ++i)
            result[i] = !data[i];
        return result;
    }

    PARAS_KERNEL_HD
    marray operator~() const noexcept {
        marray result;
        for (std::size_t i = 0; i < N; ++i) {
            result.data[i] = ~data[i];
        }
        return result;
    }

    PARAS_KERNEL_HD
    marray& operator++() noexcept {
        for (std::size_t i = 0; i < N; ++i)
            ++data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray operator++(int) noexcept {
        marray tmp(*this);
        ++(*this);
        return tmp;
    }
    PARAS_KERNEL_HD
    marray& operator--() noexcept {
        for (std::size_t i = 0; i < N; ++i)
            --data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray operator--(int) noexcept {
        marray tmp(*this);
        --(*this);
        return tmp;
    }
};

using mfloat2 = marray<float, 2>;
using mfloat3 = marray<float, 3>;
using mfloat4 = marray<float, 4>;
using mfloat8 = marray<float, 8>;
using mfloat16 = marray<float, 16>;

using mdouble2 = marray<double, 2>;
using mdouble3 = marray<double, 3>;
using mdouble4 = marray<double, 4>;
using mdouble8 = marray<double, 8>;
using mdouble16 = marray<double, 16>;

using mbool2 = marray<bool, 2>;
using mbool3 = marray<bool, 3>;
using mbool4 = marray<bool, 4>;
using mbool8 = marray<bool, 8>;
using mbool16 = marray<bool, 16>;

using mchar2 = marray<int8_t, 2>;
using mchar3 = marray<int8_t, 3>;
using mchar4 = marray<int8_t, 4>;
using mchar8 = marray<int8_t, 8>;
using mchar16 = marray<int8_t, 16>;

using muchar2 = marray<uint8_t, 2>;
using muchar3 = marray<uint8_t, 3>;
using muchar4 = marray<uint8_t, 4>;
using muchar8 = marray<uint8_t, 8>;
using muchar16 = marray<uint8_t, 16>;

using mshort2 = marray<int16_t, 2>;
using mshort3 = marray<int16_t, 3>;
using mshort4 = marray<int16_t, 4>;
using mshort8 = marray<int16_t, 8>;
using mshort16 = marray<int16_t, 16>;

using mushort2 = marray<uint16_t, 2>;
using mushort3 = marray<uint16_t, 3>;
using mushort4 = marray<uint16_t, 4>;
using mushort8 = marray<uint16_t, 8>;
using mushort16 = marray<uint16_t, 16>;

using mint2 = marray<int32_t, 2>;
using mint3 = marray<int32_t, 3>;
using mint4 = marray<int32_t, 4>;
using mint8 = marray<int32_t, 8>;
using mint16 = marray<int32_t, 16>;

using muint2 = marray<uint32_t, 2>;
using muint3 = marray<uint32_t, 3>;
using muint4 = marray<uint32_t, 4>;
using muint8 = marray<uint32_t, 8>;
using muint16 = marray<uint32_t, 16>;

using mlong2 = marray<int64_t, 2>;
using mlong3 = marray<int64_t, 3>;
using mlong4 = marray<int64_t, 4>;
using mlong8 = marray<int64_t, 8>;
using mlong16 = marray<int64_t, 16>;

using mulong2 = marray<uint64_t, 2>;
using mulong3 = marray<uint64_t, 3>;
using mulong4 = marray<uint64_t, 4>;
using mulong8 = marray<uint64_t, 8>;
using mulong16 = marray<uint64_t, 16>;

using mhalf2 = marray<half, 2>;
using mhalf3 = marray<half, 3>;
using mhalf4 = marray<half, 4>;
using mhalf8 = marray<half, 8>;
using mhalf16 = marray<half, 16>;

using char2 = vec<int8_t, 2>;
using char3 = vec<int8_t, 3>;
using char4 = vec<int8_t, 4>;
using char8 = vec<int8_t, 8>;
using char16 = vec<int8_t, 16>;

using uchar2 = vec<uint8_t, 2>;
using uchar3 = vec<uint8_t, 3>;
using uchar4 = vec<uint8_t, 4>;
using uchar8 = vec<uint8_t, 8>;
using uchar16 = vec<uint8_t, 16>;

using short2 = vec<int16_t, 2>;
using short3 = vec<int16_t, 3>;
using short4 = vec<int16_t, 4>;
using short8 = vec<int16_t, 8>;
using short16 = vec<int16_t, 16>;

using ushort2 = vec<uint16_t, 2>;
using ushort3 = vec<uint16_t, 3>;
using ushort4 = vec<uint16_t, 4>;
using ushort8 = vec<uint16_t, 8>;
using ushort16 = vec<uint16_t, 16>;

using int2 = vec<int32_t, 2>;
using int3 = vec<int32_t, 3>;
using int4 = vec<int32_t, 4>;
using int8 = vec<int32_t, 8>;
using int16 = vec<int32_t, 16>;

using uint2 = vec<uint32_t, 2>;
using uint3 = vec<uint32_t, 3>;
using uint4 = vec<uint32_t, 4>;
using uint8 = vec<uint32_t, 8>;
using uint16 = vec<uint32_t, 16>;

using long2 = vec<int64_t, 2>;
using long3 = vec<int64_t, 3>;
using long4 = vec<int64_t, 4>;
using long8 = vec<int64_t, 8>;
using long16 = vec<int64_t, 16>;

using ulong2 = vec<uint64_t, 2>;
using ulong3 = vec<uint64_t, 3>;
using ulong4 = vec<uint64_t, 4>;
using ulong8 = vec<uint64_t, 8>;
using ulong16 = vec<uint64_t, 16>;

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
using float8 = vec<float, 8>;
using float16 = vec<float, 16>;

using double2 = vec<double, 2>;
using double3 = vec<double, 3>;
using double4 = vec<double, 4>;
using double8 = vec<double, 8>;
using double16 = vec<double, 16>;

using half2 = vec<half, 2>;
using half3 = vec<half, 3>;
using half4 = vec<half, 4>;
using half8 = vec<half, 8>;
using half16 = vec<half, 16>;

template <typename To, typename From>
PARAS_KERNEL_HD To bit_cast(const From& from) noexcept {
    static_assert(sizeof(To) == sizeof(From), "sycl::bit_cast requires equally-sized types");
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

} // namespace sycl

namespace std {

template <>
class numeric_limits<sycl::half> {
public:
    static constexpr bool is_specialized = true;

    PARAS_KERNEL_HD static sycl::half min() noexcept { return sycl::half(6.103515625e-05f); }

    PARAS_KERNEL_HD static sycl::half max() noexcept { return sycl::half(65504.0f); }

    PARAS_KERNEL_HD static sycl::half lowest() noexcept { return sycl::half(-65504.0f); }

    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;

    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;

    static constexpr int radix = 2;

    PARAS_KERNEL_HD static sycl::half epsilon() noexcept { return sycl::half(0.0009765625f); }

    PARAS_KERNEL_HD static sycl::half round_error() noexcept { return sycl::half(0.5f); }

    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;

    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;

    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;

    static constexpr float_denorm_style has_denorm = denorm_present;

    static constexpr bool has_denorm_loss = false;

    PARAS_KERNEL_HD static sycl::half infinity() noexcept { return sycl::half(__builtin_inff()); }

    PARAS_KERNEL_HD static sycl::half quiet_NaN() noexcept {
        return sycl::half(__builtin_nanf(""));
    }

    PARAS_KERNEL_HD static sycl::half signaling_NaN() noexcept {
        return sycl::half(__builtin_nansf(""));
    }

    PARAS_KERNEL_HD static sycl::half denorm_min() noexcept {
        return sycl::half(5.960464477539063e-08f);
    }

    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;

    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr float_round_style round_style = round_to_nearest;
};

} // namespace std

#endif

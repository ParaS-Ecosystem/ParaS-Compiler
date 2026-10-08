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

#ifndef __PARAS_MARRAY_HPP__
#define __PARAS_MARRAY_HPP__

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "half.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename T, std::size_t N> class marray;

namespace detail {
template <typename A> struct marray_arg_count {
  static constexpr std::size_t value = 1;
};
template <typename U, std::size_t M> struct marray_arg_count<marray<U, M>> {
  static constexpr std::size_t value = M;
};
template <typename A> struct is_marray_arg : std::false_type {};
template <typename U, std::size_t M>
struct is_marray_arg<marray<U, M>> : std::true_type {};
}

template <typename T, std::size_t N>
class marray {
    struct fill_tag {};
    struct flat_tag {};

    template <typename A, typename... R>
    PARAS_KERNEL_HD static constexpr T flat_elem(std::size_t i, const A &a,
                                                 const R &...r) {
        constexpr std::size_t c = detail::marray_arg_count<A>::value;
        if (i < c) {
            if constexpr (detail::is_marray_arg<A>::value)
                return static_cast<T>(a.data[i]);
            else
                return static_cast<T>(a);
        }
        if constexpr (sizeof...(R) > 0)
            return flat_elem(i - c, r...);
        else
            return T{};
    }

    template <std::size_t... I>
    PARAS_KERNEL_HD constexpr marray(fill_tag, std::index_sequence<I...>,
                                     const T &v)
        : data{((void)I, v)...} {}

    template <std::size_t... I, typename... Args>
    PARAS_KERNEL_HD constexpr marray(flat_tag, std::index_sequence<I...>,
                                     const Args &...args)
        : data{flat_elem(I, args...)...} {}

    template <typename A>
    static constexpr bool valid_arg =
        detail::is_marray_arg<A>::value || std::is_constructible_v<T, A>;

public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;

    T data[N];

    PARAS_KERNEL_HD
    constexpr marray() noexcept : data{} {}

    PARAS_KERNEL_HD
    explicit constexpr marray(const T& v) noexcept
        : marray(fill_tag{}, std::make_index_sequence<N>{}, v) {}

    template <typename... Args,
              typename = std::enable_if_t<
                  (detail::marray_arg_count<Args>::value + ... + 0) == N &&
                  (valid_arg<Args> && ...) &&
                  !(sizeof...(Args) == 1 &&
                    (std::is_same_v<Args, marray> && ...))>>
    PARAS_KERNEL_HD
    constexpr marray(const Args &...args) noexcept
        : marray(flat_tag{}, std::make_index_sequence<N>{}, args...) {}

    PARAS_KERNEL_HD constexpr marray(const marray &) = default;
    PARAS_KERNEL_HD constexpr marray(marray &&) = default;
    PARAS_KERNEL_HD marray &operator=(const marray &) = default;
    PARAS_KERNEL_HD marray &operator=(marray &&) = default;

    PARAS_KERNEL_HD
    marray &operator=(const T &v) noexcept {
        for (std::size_t i = 0; i < N; ++i)
            data[i] = v;
        return *this;
    }

    template <std::size_t M = N, typename = std::enable_if_t<M == 1>>
    PARAS_KERNEL_HD constexpr operator T() const noexcept { return data[0]; }

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
        for (std::size_t i = 0; i < N; ++i) data[i] += rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
	marray& operator+=(const T& rhs) noexcept {
	    for (std::size_t i = 0; i < N; ++i) data[i] += rhs;
        return *this;
		}

    PARAS_KERNEL_HD
    marray& operator-=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] -= rhs.data[i];
        return *this;
    }

	PARAS_KERNEL_HD
    marray& operator-=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] -= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator*=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] *= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator*=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] *= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator/=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] /= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator/=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] /= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator%=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] %= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator%=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] %= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator&=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] &= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator&=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] &= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator|=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] |= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator|=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] |= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator^=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] ^= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator^=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] ^= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator<<=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] <<= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator<<=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] <<= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray& operator>>=(const marray& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] >>= rhs.data[i];
        return *this;
    }
	PARAS_KERNEL_HD
    marray& operator>>=(const T& rhs) noexcept {
        for (std::size_t i = 0; i < N; ++i) data[i] >>= rhs;
        return *this;
    }

    PARAS_KERNEL_HD
    marray operator+() const noexcept { return *this; }
    PARAS_KERNEL_HD
    marray operator-() const noexcept {
        marray result;
        for (std::size_t i = 0; i < N; ++i) result.data[i] = -data[i];
        return result;
    }

    PARAS_KERNEL_HD
    marray<bool, N> operator!() const noexcept {
        marray<bool, N> result{};
        for (std::size_t i = 0; i < N; ++i) result[i] = !data[i];
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
        for (std::size_t i = 0; i < N; ++i) ++data[i];
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
        for (std::size_t i = 0; i < N; ++i) --data[i];
        return *this;
    }
    PARAS_KERNEL_HD
    marray operator--(int) noexcept {
        marray tmp(*this);
        --(*this);
        return tmp;
    }
};

using mfloat2  = marray<float, 2>;
using mfloat3  = marray<float, 3>;
using mfloat4  = marray<float, 4>;
using mfloat8  = marray<float, 8>;
using mfloat16 = marray<float, 16>;

using mdouble2  = marray<double, 2>;
using mdouble3  = marray<double, 3>;
using mdouble4  = marray<double, 4>;
using mdouble8  = marray<double, 8>;
using mdouble16 = marray<double, 16>;

using mbool2 = marray<bool, 2>;   using mbool3 = marray<bool, 3>;
using mbool4 = marray<bool, 4>;   using mbool8 = marray<bool, 8>;
using mbool16 = marray<bool, 16>;

using mchar2 = marray<int8_t, 2>;  using mchar3 = marray<int8_t, 3>;
using mchar4 = marray<int8_t, 4>;  using mchar8 = marray<int8_t, 8>;
using mchar16 = marray<int8_t, 16>;

using muchar2 = marray<uint8_t, 2>;  using muchar3 = marray<uint8_t, 3>;
using muchar4 = marray<uint8_t, 4>;  using muchar8 = marray<uint8_t, 8>;
using muchar16 = marray<uint8_t, 16>;

using mshort2 = marray<int16_t, 2>;  using mshort3 = marray<int16_t, 3>;
using mshort4 = marray<int16_t, 4>;  using mshort8 = marray<int16_t, 8>;
using mshort16 = marray<int16_t, 16>;

using mushort2 = marray<uint16_t, 2>;  using mushort3 = marray<uint16_t, 3>;
using mushort4 = marray<uint16_t, 4>;  using mushort8 = marray<uint16_t, 8>;
using mushort16 = marray<uint16_t, 16>;

using mint2 = marray<int32_t, 2>;  using mint3 = marray<int32_t, 3>;
using mint4 = marray<int32_t, 4>;  using mint8 = marray<int32_t, 8>;
using mint16 = marray<int32_t, 16>;

using muint2 = marray<uint32_t, 2>;  using muint3 = marray<uint32_t, 3>;
using muint4 = marray<uint32_t, 4>;  using muint8 = marray<uint32_t, 8>;
using muint16 = marray<uint32_t, 16>;

using mlong2 = marray<int64_t, 2>;  using mlong3 = marray<int64_t, 3>;
using mlong4 = marray<int64_t, 4>;  using mlong8 = marray<int64_t, 8>;
using mlong16 = marray<int64_t, 16>;

using mulong2 = marray<uint64_t, 2>;  using mulong3 = marray<uint64_t, 3>;
using mulong4 = marray<uint64_t, 4>;  using mulong8 = marray<uint64_t, 8>;
using mulong16 = marray<uint64_t, 16>;

using mhalf2 = marray<half, 2>;  using mhalf3 = marray<half, 3>;
using mhalf4 = marray<half, 4>;  using mhalf8 = marray<half, 8>;
using mhalf16 = marray<half, 16>;

}

#endif

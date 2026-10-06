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

#ifndef __PARAS_STREAM_HPP__
#define __PARAS_STREAM_HPP__

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <type_traits>
#include "kem_gpu/gpu_utilities.hpp"
#include "data_types.hpp"
#include "id.hpp"
#include "range.hpp"

namespace sycl {

class handler;
class property_list;

enum class stream_manipulator {
    flush,
    dec,
    hex,
    oct,
    noshowbase,
    showbase,
    noshowpos,
    showpos,
    endl,
    fixed,
    scientific,
    hexfloat,
    defaultfloat
};

inline constexpr stream_manipulator flush = stream_manipulator::flush;
inline constexpr stream_manipulator dec = stream_manipulator::dec;
inline constexpr stream_manipulator hex = stream_manipulator::hex;
inline constexpr stream_manipulator oct = stream_manipulator::oct;
inline constexpr stream_manipulator noshowbase = stream_manipulator::noshowbase;
inline constexpr stream_manipulator showbase = stream_manipulator::showbase;
inline constexpr stream_manipulator noshowpos = stream_manipulator::noshowpos;
inline constexpr stream_manipulator showpos = stream_manipulator::showpos;
inline constexpr stream_manipulator endl = stream_manipulator::endl;
inline constexpr stream_manipulator fixed = stream_manipulator::fixed;
inline constexpr stream_manipulator scientific = stream_manipulator::scientific;
inline constexpr stream_manipulator hexfloat = stream_manipulator::hexfloat;
inline constexpr stream_manipulator defaultfloat = stream_manipulator::defaultfloat;

struct paras_precision_manipulator {
    int precision;
};
struct paras_width_manipulator {
    int width;
};
template <typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
paras_precision_manipulator setprecision(Int precision) {
    return {static_cast<int>(precision)};
}
template <typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
paras_width_manipulator setw(Int width) {
    return {static_cast<int>(width)};
}

class stream {
public:
    stream(std::size_t bufferSize, std::size_t workItemBufferSize, handler&,
           const property_list& = {})
        : bufferSize_(bufferSize), workItemBufferSize_(workItemBufferSize), id_(next_id()) {}

    std::size_t size() const noexcept { return bufferSize_; }
    std::size_t get_size() const noexcept { return bufferSize_; }
    std::size_t get_work_item_buffer_size() const noexcept { return workItemBufferSize_; }
    std::size_t get_max_statement_size() const noexcept { return workItemBufferSize_; }

    PARAS_KERNEL_HD const stream& operator<<(const char* s) const {
        printf("%s", s);
        return *this;
    }
    PARAS_KERNEL_HD const stream& operator<<(char c) const {
        printf("%c", c);
        return *this;
    }
    PARAS_KERNEL_HD const stream& operator<<(bool b) const {
        printf("%s", b ? "true" : "false");
        return *this;
    }
    template <typename T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool> &&
                                               !std::is_same_v<T, char>,
                                           int> = 0>
    PARAS_KERNEL_HD const stream& operator<<(T v) const {
        if (base_ == 16 || base_ == 8) {
            const unsigned long long u = static_cast<unsigned long long>(v);
            if (base_ == 16)
                printf(showbase_ ? "%#llx" : "%llx", u);
            else
                printf(showbase_ ? "%#llo" : "%llo", u);
        } else if (std::is_signed_v<T>) {
            printf("%lld", static_cast<long long>(v));
        } else {
            printf("%llu", static_cast<unsigned long long>(v));
        }
        return *this;
    }
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    PARAS_KERNEL_HD const stream& operator<<(T v) const {
        const double d = static_cast<double>(v);
        if (float_ == 1)
            printf("%.*f", precision_, d);
        else if (float_ == 2)
            printf("%.*e", precision_, d);
        else if (float_ == 3)
            printf("%a", d);
        else
            printf("%.*g", precision_, d);
        return *this;
    }
    PARAS_KERNEL_HD const stream& operator<<(sycl::half h) const {
        return *this << static_cast<float>(h);
    }
    template <typename T>
    PARAS_KERNEL_HD const stream& operator<<(T* p) const {
        printf("%p", static_cast<const void*>(p));
        return *this;
    }
    template <typename T, int N>
    PARAS_KERNEL_HD const stream& operator<<(const vec<T, N>& v) const {
        for (int i = 0; i < N; ++i) {
            if (i != 0)
                *this << ", ";
            *this << v[i];
        }
        return *this;
    }
    template <int D>
    PARAS_KERNEL_HD const stream& operator<<(const id<D>& v) const {
        return print_index(v);
    }
    template <int D>
    PARAS_KERNEL_HD const stream& operator<<(const range<D>& v) const {
        return print_index(v);
    }

    PARAS_KERNEL_HD const stream& operator<<(stream_manipulator m) const {
        switch (m) {
        case stream_manipulator::endl:
            printf("\n");
            break;
        case stream_manipulator::dec:
            base_ = 10;
            break;
        case stream_manipulator::hex:
            base_ = 16;
            break;
        case stream_manipulator::oct:
            base_ = 8;
            break;
        case stream_manipulator::showbase:
            showbase_ = true;
            break;
        case stream_manipulator::noshowbase:
            showbase_ = false;
            break;
        case stream_manipulator::fixed:
            float_ = 1;
            break;
        case stream_manipulator::scientific:
            float_ = 2;
            break;
        case stream_manipulator::hexfloat:
            float_ = 3;
            break;
        case stream_manipulator::defaultfloat:
            float_ = 0;
            break;
        default:
            break;
        }
        return *this;
    }
    PARAS_KERNEL_HD const stream& operator<<(paras_precision_manipulator p) const {
        precision_ = p.precision;
        return *this;
    }
    PARAS_KERNEL_HD const stream& operator<<(paras_width_manipulator) const { return *this; }

    bool operator==(const stream& rhs) const noexcept { return id_ == rhs.id_; }
    bool operator!=(const stream& rhs) const noexcept { return !(*this == rhs); }
    std::uint64_t get_identity() const noexcept { return id_; }

private:
    template <typename Index>
    PARAS_KERNEL_HD const stream& print_index(const Index& v) const {
        *this << "{";
        for (int i = 0; i < Index::dimensions; ++i) {
            if (i != 0)
                *this << ", ";
            *this << v[i];
        }
        return *this << "}";
    }

    static std::uint64_t next_id() {
        static std::atomic<std::uint64_t> counter{1};
        return counter.fetch_add(1);
    }

    std::size_t bufferSize_;
    std::size_t workItemBufferSize_;
    std::uint64_t id_;
    mutable int base_ = 10;
    mutable bool showbase_ = false;
    mutable int float_ = 0;
    mutable int precision_ = 6;
};

} // namespace sycl

namespace std {
template <>
struct hash<sycl::stream> {
    std::size_t operator()(const sycl::stream& s) const noexcept {
        return std::hash<std::uint64_t>{}(s.get_identity());
    }
};
} // namespace std

#endif

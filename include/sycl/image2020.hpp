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

#ifndef __PARAS_IMAGE2020_HPP__
#define __PARAS_IMAGE2020_HPP__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include "access.hpp"
#include "aspect.hpp"
#include "data_types.hpp"
#include "exception.hpp"
#include "handler.hpp"
#include "id.hpp"
#include "property_list.hpp"
#include "range.hpp"
#include "sampler.hpp"

namespace sycl {

enum class image_format {
    r8g8b8a8_unorm,
    r16g16b16a16_unorm,
    r8g8b8a8_sint,
    r16g16b16a16_sint,
    r32b32g32a32_sint,
    r8g8b8a8_uint,
    r16g16b16a16_uint,
    r32b32g32a32_uint,
    r16b16g16a16_sfloat,
    r32g32b32a32_sfloat,
    b8g8r8a8_unorm
};

enum class image_target { device, host_task };

struct image_sampler {
    addressing_mode addressing;
    coordinate_normalization_mode coordinate;
    filtering_mode filtering;
};

using image_allocator = std::allocator<std::byte>;

template <int Dimensions, typename AllocatorT>
class unsampled_image;
template <int Dimensions, typename AllocatorT>
class sampled_image;
template <typename DataT, int Dimensions, access_mode AccessMode, image_target AccessTarget>
class unsampled_image_accessor;
template <typename DataT, int Dimensions, access_mode AccessMode>
class host_unsampled_image_accessor;
template <typename DataT, int Dimensions, image_target AccessTarget>
class sampled_image_accessor;
template <typename DataT, int Dimensions>
class host_sampled_image_accessor;

namespace detail {
namespace paras_img {

enum class kind { unorm, sint, uint, sfloat };
struct format_info {
    kind k;
    int bits;
    bool bgra;
};
inline format_info info(image_format f) {
    switch (f) {
    case image_format::r8g8b8a8_unorm: return {kind::unorm, 8, false};
    case image_format::r16g16b16a16_unorm: return {kind::unorm, 16, false};
    case image_format::r8g8b8a8_sint: return {kind::sint, 8, false};
    case image_format::r16g16b16a16_sint: return {kind::sint, 16, false};
    case image_format::r32b32g32a32_sint: return {kind::sint, 32, false};
    case image_format::r8g8b8a8_uint: return {kind::uint, 8, false};
    case image_format::r16g16b16a16_uint: return {kind::uint, 16, false};
    case image_format::r32b32g32a32_uint: return {kind::uint, 32, false};
    case image_format::r16b16g16a16_sfloat: return {kind::sfloat, 16, false};
    case image_format::r32g32b32a32_sfloat: return {kind::sfloat, 32, false};
    case image_format::b8g8r8a8_unorm: return {kind::unorm, 8, true};
    }
    return {kind::unorm, 8, false};
}
inline std::size_t elem_size(image_format f) { return static_cast<std::size_t>(info(f).bits) / 2; }

template <typename T>
inline constexpr bool valid_data_t =
    std::is_same_v<std::remove_const_t<T>, vec<std::int32_t, 4>> ||
    std::is_same_v<std::remove_const_t<T>, vec<std::uint32_t, 4>> ||
    std::is_same_v<std::remove_const_t<T>, vec<float, 4>> ||
    std::is_same_v<std::remove_const_t<T>, vec<half, 4>>;

struct state {
    image_format format{};
    int dims = 1;
    std::size_t range[3] = {1, 1, 1};
    std::size_t pitch[2] = {0, 0};
    std::byte* data = nullptr;
    std::size_t bytes = 0;
    std::function<void(std::byte*, std::size_t)> release;
    std::function<void(const std::byte*, std::size_t)> final_data;
    bool write_back = true;
    property_list props;
    image_sampler sampler{addressing_mode::none, coordinate_normalization_mode::unnormalized,
                          filtering_mode::nearest};

    ~state() {
        if (write_back && final_data && data)
            final_data(data, bytes);
        if (release)
            release(data, bytes);
    }
    std::size_t count() const { return range[0] * range[1] * range[2]; }
    std::byte* texel(std::size_t x, std::size_t y, std::size_t z) const {
        return data + x * elem_size(format) + y * pitch[0] + z * pitch[1];
    }
};

inline void decode(const state& s, const std::byte* p, double out[4]) {
    const format_info fi = info(s.format);
    double c[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        const std::byte* q = p + i * (fi.bits / 8);
        switch (fi.k) {
        case kind::unorm:
            if (fi.bits == 8) {
                std::uint8_t v;
                std::memcpy(&v, q, 1);
                c[i] = v / 255.0;
            } else {
                std::uint16_t v;
                std::memcpy(&v, q, 2);
                c[i] = v / 65535.0;
            }
            break;
        case kind::sint:
            if (fi.bits == 8) { std::int8_t v; std::memcpy(&v, q, 1); c[i] = v; }
            else if (fi.bits == 16) { std::int16_t v; std::memcpy(&v, q, 2); c[i] = v; }
            else { std::int32_t v; std::memcpy(&v, q, 4); c[i] = v; }
            break;
        case kind::uint:
            if (fi.bits == 8) { std::uint8_t v; std::memcpy(&v, q, 1); c[i] = v; }
            else if (fi.bits == 16) { std::uint16_t v; std::memcpy(&v, q, 2); c[i] = v; }
            else { std::uint32_t v; std::memcpy(&v, q, 4); c[i] = v; }
            break;
        case kind::sfloat:
            if (fi.bits == 16) { half v; std::memcpy(&v, q, 2); c[i] = static_cast<float>(v); }
            else { float v; std::memcpy(&v, q, 4); c[i] = v; }
            break;
        }
    }
    if (fi.bgra) { out[0] = c[2]; out[1] = c[1]; out[2] = c[0]; out[3] = c[3]; }
    else { for (int i = 0; i < 4; ++i) out[i] = c[i]; }
}
template <typename I>
inline I sat(double v) {
    const double lo = static_cast<double>(std::numeric_limits<I>::lowest());
    const double hi = static_cast<double>(std::numeric_limits<I>::max());
    if (std::isnan(v)) return I(0);
    return static_cast<I>(v < lo ? lo : (v > hi ? hi : std::nearbyint(v)));
}
inline void encode(const state& s, std::byte* p, const double in[4]) {
    const format_info fi = info(s.format);
    double c[4];
    if (fi.bgra) { c[0] = in[2]; c[1] = in[1]; c[2] = in[0]; c[3] = in[3]; }
    else { for (int i = 0; i < 4; ++i) c[i] = in[i]; }
    for (int i = 0; i < 4; ++i) {
        std::byte* q = p + i * (fi.bits / 8);
        switch (fi.k) {
        case kind::unorm: {
            double v = std::isnan(c[i]) ? 0.0 : (c[i] < 0 ? 0.0 : (c[i] > 1 ? 1.0 : c[i]));
            if (fi.bits == 8) { auto u = static_cast<std::uint8_t>(std::nearbyint(v * 255.0)); std::memcpy(q, &u, 1); }
            else { auto u = static_cast<std::uint16_t>(std::nearbyint(v * 65535.0)); std::memcpy(q, &u, 2); }
            break;
        }
        case kind::sint:
            if (fi.bits == 8) { auto v = sat<std::int8_t>(c[i]); std::memcpy(q, &v, 1); }
            else if (fi.bits == 16) { auto v = sat<std::int16_t>(c[i]); std::memcpy(q, &v, 2); }
            else { auto v = sat<std::int32_t>(c[i]); std::memcpy(q, &v, 4); }
            break;
        case kind::uint:
            if (fi.bits == 8) { auto v = sat<std::uint8_t>(c[i]); std::memcpy(q, &v, 1); }
            else if (fi.bits == 16) { auto v = sat<std::uint16_t>(c[i]); std::memcpy(q, &v, 2); }
            else { auto v = sat<std::uint32_t>(c[i]); std::memcpy(q, &v, 4); }
            break;
        case kind::sfloat:
            if (fi.bits == 16) { half v(static_cast<float>(c[i])); std::memcpy(q, &v, 2); }
            else { float v = static_cast<float>(c[i]); std::memcpy(q, &v, 4); }
            break;
        }
    }
}

template <typename DataT>
inline DataT to_data(const double c[4]) {
    using E = typename std::remove_const_t<DataT>::element_type;
    std::remove_const_t<DataT> r;
    for (int i = 0; i < 4; ++i)
        r[i] = static_cast<E>(c[i]);
    return r;
}
template <typename DataT>
inline void from_data(const DataT& v, double c[4]) {
    for (int i = 0; i < 4; ++i)
        c[i] = static_cast<double>(v[i]);
}

template <int D, typename CoordT>
inline void int_coords(const CoordT& c, long long out[3]) {
    out[0] = out[1] = out[2] = 0;
    if constexpr (D == 1) out[0] = static_cast<long long>(c);
    else
        for (int i = 0; i < D; ++i)
            out[i] = static_cast<long long>(c[i]);
}
template <int D, typename CoordT>
inline void float_coords(const CoordT& c, double out[3]) {
    out[0] = out[1] = out[2] = 0;
    if constexpr (D == 1) out[0] = static_cast<double>(c);
    else
        for (int i = 0; i < D; ++i)
            out[i] = static_cast<double>(c[i]);
}
template <int D, typename CoordT>
inline constexpr bool int_coord_ok =
    (D == 1 && std::is_same_v<CoordT, int>) || (D == 2 && std::is_same_v<CoordT, vec<int, 2>>) ||
    (D == 3 && std::is_same_v<CoordT, vec<int, 4>>);
template <int D, typename CoordT>
inline constexpr bool float_coord_ok =
    (D == 1 && std::is_same_v<CoordT, float>) ||
    (D == 2 && std::is_same_v<CoordT, vec<float, 2>>) ||
    (D == 3 && std::is_same_v<CoordT, vec<float, 4>>);

inline void read_texel(const state& s, const long long xyz[3], double out[4]) {
    std::size_t p[3];
    for (int d = 0; d < 3; ++d) {
        long long v = xyz[d] < 0 ? 0 : xyz[d];
        p[d] = static_cast<std::size_t>(v) >= s.range[d] ? s.range[d] - 1 : static_cast<std::size_t>(v);
    }
    decode(s, s.texel(p[0], p[1], p[2]), out);
}
inline void write_texel(const state& s, const long long xyz[3], const double in[4]) {
    for (int d = 0; d < 3; ++d)
        if (xyz[d] < 0 || static_cast<std::size_t>(xyz[d]) >= s.range[d])
            return;
    encode(s, s.texel(static_cast<std::size_t>(xyz[0]), static_cast<std::size_t>(xyz[1]),
                      static_cast<std::size_t>(xyz[2])),
           in);
}

inline void sample(const state& s, const double coords[3], bool allow_linear, double out[4]) {
    const image_sampler sm = s.sampler;
    const bool norm = sm.coordinate == coordinate_normalization_mode::normalized;
    const bool linear = allow_linear && sm.filtering == filtering_mode::linear;
    long long i0[3] = {0, 0, 0}, i1[3] = {0, 0, 0};
    double a[3] = {0, 0, 0};
    bool border0[3] = {false, false, false}, border1[3] = {false, false, false};
    for (int d = 0; d < s.dims; ++d) {
        const long long w = static_cast<long long>(s.range[d]);
        const double sc = coords[d];
        double u = norm ? sc * static_cast<double>(w) : sc;
        if (sm.addressing == addressing_mode::repeat && norm) {
            u = (sc - std::floor(sc)) * static_cast<double>(w);
            if (!linear) {
                long long i = static_cast<long long>(std::floor(u));
                i0[d] = i > w - 1 ? i - w : i;
            } else {
                i0[d] = static_cast<long long>(std::floor(u - 0.5));
                i1[d] = i0[d] + 1;
                if (i0[d] < 0) i0[d] += w;
                if (i1[d] > w - 1) i1[d] -= w;
                a[d] = (u - 0.5) - std::floor(u - 0.5);
            }
            continue;
        }
        if (sm.addressing == addressing_mode::mirrored_repeat && norm) {
            double sp = 2.0 * std::nearbyint(0.5 * sc);
            sp = std::fabs(sc - sp);
            u = sp * static_cast<double>(w);
            if (!linear) {
                long long i = static_cast<long long>(std::floor(u));
                i0[d] = i < w - 1 ? i : w - 1;
            } else {
                i0[d] = static_cast<long long>(std::floor(u - 0.5));
                i1[d] = i0[d] + 1;
                if (i0[d] < 0) i0[d] = 0;
                if (i1[d] > w - 1) i1[d] = w - 1;
                a[d] = (u - 0.5) - std::floor(u - 0.5);
            }
            continue;
        }
        auto fix = [&](long long i, bool& border) {
            if (i >= 0 && i < w) return i;
            if (sm.addressing == addressing_mode::clamp) border = true;
            return i < 0 ? 0LL : w - 1;
        };
        if (!linear) {
            i0[d] = fix(static_cast<long long>(std::floor(u)), border0[d]);
        } else {
            const long long j = static_cast<long long>(std::floor(u - 0.5));
            i0[d] = fix(j, border0[d]);
            i1[d] = fix(j + 1, border1[d]);
            a[d] = (u - 0.5) - std::floor(u - 0.5);
        }
    }
    auto fetch = [&](const long long* idx, bool border, double* c) {
        if (border) {
            for (int k = 0; k < 4; ++k) c[k] = 0.0;
            return;
        }
        read_texel(s, idx, c);
    };
    if (!linear) {
        fetch(i0, border0[0] || border0[1] || border0[2], out);
        return;
    }
    for (int k = 0; k < 4; ++k) out[k] = 0.0;
    const int corners = 1 << s.dims;
    for (int m = 0; m < corners; ++m) {
        long long idx[3] = {0, 0, 0};
        bool border = false;
        double wgt = 1.0;
        for (int d = 0; d < s.dims; ++d) {
            const bool hi = (m >> d) & 1;
            idx[d] = hi ? i1[d] : i0[d];
            border = border || (hi ? border1[d] : border0[d]);
            wgt *= hi ? a[d] : 1.0 - a[d];
        }
        double c[4];
        fetch(idx, border, c);
        for (int k = 0; k < 4; ++k) out[k] += wgt * c[k];
    }
}

template <int Dimensions, typename AllocatorT>
inline std::shared_ptr<state> make_state(image_format fmt, const range<Dimensions>& r,
                                         const std::size_t* userPitch, AllocatorT alloc,
                                         const property_list& props) {
    auto st = std::make_shared<state>();
    st->format = fmt;
    st->dims = Dimensions;
    for (int d = 0; d < Dimensions; ++d)
        st->range[d] = r[d];
    const std::size_t es = elem_size(fmt);
    st->pitch[0] = (Dimensions > 1 && userPitch) ? userPitch[0] : st->range[0] * es;
    st->pitch[1] = (Dimensions > 2 && userPitch) ? userPitch[1] : st->pitch[0] * st->range[1];
    st->bytes = Dimensions == 1 ? st->range[0] * es
                                : (Dimensions == 2 ? st->pitch[0] * st->range[1]
                                                   : st->pitch[1] * st->range[2]);
    st->props = props;
    return st;
}
template <typename AllocatorT>
inline void allocate(state& st, AllocatorT alloc) {
    using traits = std::allocator_traits<AllocatorT>;
    st.data = reinterpret_cast<std::byte*>(traits::allocate(alloc, st.bytes ? st.bytes : 1));
    std::memset(st.data, 0, st.bytes);
    st.release = [alloc](std::byte* p, std::size_t n) mutable {
        traits::deallocate(alloc, reinterpret_cast<typename traits::pointer>(p), n ? n : 1);
    };
}
template <typename AllocatorT>
inline void attach_host(state& st, const void* host, AllocatorT alloc) {
    if (host && st.props.template paras_has_property<property::image::use_host_ptr>()) {
        st.data = static_cast<std::byte*>(const_cast<void*>(host));
        return;
    }
    allocate(st, alloc);
    if (host)
        std::memcpy(st.data, host, st.bytes);
}

template <typename Destination>
inline void set_final(state& st, Destination d) {
    if constexpr (std::is_same_v<Destination, std::nullptr_t>) {
        st.final_data = nullptr;
    } else if constexpr (std::is_pointer_v<Destination>) {
        if (d == nullptr) {
            st.final_data = nullptr;
        } else {
            void* dst = const_cast<void*>(static_cast<const void*>(d));
            st.final_data = [dst](const std::byte* src, std::size_t n) {
                if (src != dst)
                    std::memcpy(dst, src, n);
            };
        }
    } else {
        st.final_data = [d](const std::byte* src, std::size_t n) {
            if (auto sp = d.lock())
                std::memcpy(static_cast<void*>(sp.get()), src, n);
        };
    }
}

}
}

#define PARAS_IMG_PROPERTY_INTERFACE                                                               \
    template <typename Property>                                                                   \
    bool has_property() const noexcept {                                                           \
        return st_->props.template paras_has_property<Property>();                                 \
    }                                                                                              \
    template <typename Property>                                                                   \
    Property get_property() const {                                                                \
        return st_->props.template paras_get_property<Property>();                                 \
    }

template <int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
class unsampled_image {
    static_assert(Dimensions >= 1 && Dimensions <= 3, "image dimensions must be 1, 2 or 3");
    using st_t = detail::paras_img::state;

    void init(const void* host, image_format fmt, const range<Dimensions>& r,
              const std::size_t* pitch, AllocatorT alloc, const property_list& props) {
        st_ = detail::paras_img::make_state(fmt, r, pitch, alloc, props);
        alloc_ = alloc;
        if (host) {
            detail::paras_img::attach_host(*st_, host, alloc);
            if (st_->data != host)
                detail::paras_img::set_final(*st_, const_cast<void*>(host));
        } else {
            detail::paras_img::allocate(*st_, alloc);
        }
    }
    template <int D>
    static const std::size_t* pitch_ptr(const range<D>& p, std::size_t (&buf)[2]) {
        for (int i = 0; i < D; ++i)
            buf[i] = p[i];
        return buf;
    }

public:
    unsampled_image(image_format format, const range<Dimensions>& rangeRef,
                    const property_list& propList = {}) {
        init(nullptr, format, rangeRef, nullptr, AllocatorT{}, propList);
    }
    unsampled_image(image_format format, const range<Dimensions>& rangeRef, AllocatorT allocator,
                    const property_list& propList = {}) {
        init(nullptr, format, rangeRef, nullptr, allocator, propList);
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    unsampled_image(image_format format, const range<Dimensions>& rangeRef,
                    const range<D - 1>& pitch, const property_list& propList = {}) {
        std::size_t b[2];
        init(nullptr, format, rangeRef, pitch_ptr(pitch, b), AllocatorT{}, propList);
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    unsampled_image(image_format format, const range<Dimensions>& rangeRef,
                    const range<D - 1>& pitch, AllocatorT allocator,
                    const property_list& propList = {}) {
        std::size_t b[2];
        init(nullptr, format, rangeRef, pitch_ptr(pitch, b), allocator, propList);
    }
    unsampled_image(void* hostPointer, image_format format, const range<Dimensions>& rangeRef,
                    const property_list& propList = {}) {
        init(hostPointer, format, rangeRef, nullptr, AllocatorT{}, propList);
    }
    unsampled_image(void* hostPointer, image_format format, const range<Dimensions>& rangeRef,
                    AllocatorT allocator, const property_list& propList = {}) {
        init(hostPointer, format, rangeRef, nullptr, allocator, propList);
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    unsampled_image(void* hostPointer, image_format format, const range<Dimensions>& rangeRef,
                    const range<D - 1>& pitch, const property_list& propList = {}) {
        std::size_t b[2];
        init(hostPointer, format, rangeRef, pitch_ptr(pitch, b), AllocatorT{}, propList);
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    unsampled_image(void* hostPointer, image_format format, const range<Dimensions>& rangeRef,
                    const range<D - 1>& pitch, AllocatorT allocator,
                    const property_list& propList = {}) {
        std::size_t b[2];
        init(hostPointer, format, rangeRef, pitch_ptr(pitch, b), allocator, propList);
    }
    unsampled_image(std::shared_ptr<void>& hostPointer, image_format format,
                    const range<Dimensions>& rangeRef, const property_list& propList = {})
        : unsampled_image(hostPointer.get(), format, rangeRef, propList) {
        paras_keep_shared(hostPointer);
    }
    unsampled_image(std::shared_ptr<void>& hostPointer, image_format format,
                    const range<Dimensions>& rangeRef, AllocatorT allocator,
                    const property_list& propList = {})
        : unsampled_image(hostPointer.get(), format, rangeRef, allocator, propList) {
        paras_keep_shared(hostPointer);
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    unsampled_image(std::shared_ptr<void>& hostPointer, image_format format,
                    const range<Dimensions>& rangeRef, const range<D - 1>& pitch,
                    const property_list& propList = {})
        : unsampled_image(hostPointer.get(), format, rangeRef, pitch, propList) {
        paras_keep_shared(hostPointer);
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    unsampled_image(std::shared_ptr<void>& hostPointer, image_format format,
                    const range<Dimensions>& rangeRef, const range<D - 1>& pitch,
                    AllocatorT allocator, const property_list& propList = {})
        : unsampled_image(hostPointer.get(), format, rangeRef, pitch, allocator, propList) {
        paras_keep_shared(hostPointer);
    }

    PARAS_IMG_PROPERTY_INTERFACE

    range<Dimensions> get_range() const {
        range<Dimensions> r;
        for (int d = 0; d < Dimensions; ++d)
            r[d] = st_->range[d];
        return r;
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    range<D - 1> get_pitch() const {
        range<D - 1> p;
        for (int d = 0; d < D - 1; ++d)
            p[d] = st_->pitch[d];
        return p;
    }
    std::size_t byte_size() const noexcept { return st_->bytes; }
    std::size_t size() const noexcept { return st_->count(); }
    AllocatorT get_allocator() const { return alloc_; }

    template <typename DataT,
              access_mode Mode =
                  (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write),
              image_target Targ = image_target::device>
    unsampled_image_accessor<DataT, Dimensions, Mode, Targ>
    get_access(handler& commandGroupHandler, const property_list& propList = {}) {
        return unsampled_image_accessor<DataT, Dimensions, Mode, Targ>(*this, commandGroupHandler,
                                                                       propList);
    }
    template <typename DataT, access_mode Mode = (std::is_const_v<DataT> ? access_mode::read
                                                                         : access_mode::read_write)>
    host_unsampled_image_accessor<DataT, Dimensions, Mode>
    get_host_access(const property_list& propList = {}) {
        return host_unsampled_image_accessor<DataT, Dimensions, Mode>(*this, propList);
    }

    template <typename Destination = std::nullptr_t>
    void set_final_data(Destination finalData = nullptr) {
        detail::paras_img::set_final(*st_, finalData);
    }
    void set_write_back(bool flag = true) { st_->write_back = flag; }

    friend bool operator==(const unsampled_image& a, const unsampled_image& b) {
        return a.st_ == b.st_;
    }
    friend bool operator!=(const unsampled_image& a, const unsampled_image& b) { return !(a == b); }

    const std::shared_ptr<st_t>& paras_state() const noexcept { return st_; }

private:
    void paras_keep_shared(std::shared_ptr<void>& sp) {
        auto keep = sp;
        auto prev = st_->final_data;
        st_->final_data = [keep, prev](const std::byte* src, std::size_t n) {
            if (prev) prev(src, n);
        };
    }
    std::shared_ptr<st_t> st_;
    AllocatorT alloc_{};
};

template <int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
class sampled_image {
    static_assert(Dimensions >= 1 && Dimensions <= 3, "image dimensions must be 1, 2 or 3");
    using st_t = detail::paras_img::state;

    void init(const void* host, image_format fmt, image_sampler sampler,
              const range<Dimensions>& r, const std::size_t* pitch, const property_list& props) {
        st_ = detail::paras_img::make_state(fmt, r, pitch, AllocatorT{}, props);
        st_->sampler = sampler;
        detail::paras_img::attach_host(*st_, host, AllocatorT{});
    }

public:
    sampled_image(const void* hostPointer, image_format format, image_sampler sampler,
                  const range<Dimensions>& rangeRef, const property_list& propList = {}) {
        init(hostPointer, format, sampler, rangeRef, nullptr, propList);
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    sampled_image(const void* hostPointer, image_format format, image_sampler sampler,
                  const range<Dimensions>& rangeRef, const range<D - 1>& pitch,
                  const property_list& propList = {}) {
        std::size_t b[2];
        for (int i = 0; i < D - 1; ++i)
            b[i] = pitch[i];
        init(hostPointer, format, sampler, rangeRef, b, propList);
    }
    sampled_image(std::shared_ptr<const void>& hostPointer, image_format format,
                  image_sampler sampler, const range<Dimensions>& rangeRef,
                  const property_list& propList = {})
        : sampled_image(hostPointer.get(), format, sampler, rangeRef, propList) {
        keep_ = hostPointer;
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    sampled_image(std::shared_ptr<const void>& hostPointer, image_format format,
                  image_sampler sampler, const range<Dimensions>& rangeRef,
                  const range<D - 1>& pitch, const property_list& propList = {})
        : sampled_image(hostPointer.get(), format, sampler, rangeRef, pitch, propList) {
        keep_ = hostPointer;
    }

    PARAS_IMG_PROPERTY_INTERFACE

    range<Dimensions> get_range() const {
        range<Dimensions> r;
        for (int d = 0; d < Dimensions; ++d)
            r[d] = st_->range[d];
        return r;
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>>
    range<D - 1> get_pitch() const {
        range<D - 1> p;
        for (int d = 0; d < D - 1; ++d)
            p[d] = st_->pitch[d];
        return p;
    }
    std::size_t byte_size() const noexcept { return st_->bytes; }
    std::size_t size() const noexcept { return st_->count(); }

    template <typename DataT, image_target Targ = image_target::device>
    sampled_image_accessor<DataT, Dimensions, Targ> get_access(handler& commandGroupHandler,
                                                               const property_list& propList = {}) {
        return sampled_image_accessor<DataT, Dimensions, Targ>(*this, commandGroupHandler,
                                                               propList);
    }
    template <typename DataT>
    host_sampled_image_accessor<DataT, Dimensions>
    get_host_access(const property_list& propList = {}) {
        return host_sampled_image_accessor<DataT, Dimensions>(*this, propList);
    }

    friend bool operator==(const sampled_image& a, const sampled_image& b) { return a.st_ == b.st_; }
    friend bool operator!=(const sampled_image& a, const sampled_image& b) { return !(a == b); }

    const std::shared_ptr<st_t>& paras_state() const noexcept { return st_; }

private:
    std::shared_ptr<st_t> st_;
    std::shared_ptr<const void> keep_;
};

#undef PARAS_IMG_PROPERTY_INTERFACE

namespace detail {
namespace paras_img {
template <typename DataT, int Dimensions>
class acc_base {
    static_assert(valid_data_t<DataT>, "image accessor DataT must be int4, uint4, float4 or half4");

public:
    std::size_t size() const noexcept { return st_ ? st_->count() : 0; }
    template <typename Property>
    bool has_property() const noexcept {
        return props_.template paras_has_property<Property>();
    }
    template <typename Property>
    Property get_property() const {
        return props_.template paras_get_property<Property>();
    }
    const std::shared_ptr<state>& paras_state() const noexcept { return st_; }

protected:
    acc_base() = default;
    acc_base(std::shared_ptr<state> st, const property_list& props) : st_(std::move(st)), props_(props) {}
    template <typename CoordT>
    DataT do_read_int(const CoordT& coords) const {
        long long xyz[3];
        int_coords<Dimensions>(coords, xyz);
        double c[4];
        read_texel(*st_, xyz, c);
        return to_data<DataT>(c);
    }
    template <typename CoordT>
    void do_write(const CoordT& coords, const DataT& color) const {
        long long xyz[3];
        int_coords<Dimensions>(coords, xyz);
        double c[4];
        from_data(color, c);
        write_texel(*st_, xyz, c);
    }
    template <typename CoordT>
    DataT do_sample(const CoordT& coords) const {
        double xyz[3];
        float_coords<Dimensions>(coords, xyz);
        double c[4];
        constexpr bool fp = std::is_same_v<std::remove_const_t<DataT>, vec<float, 4>> ||
                            std::is_same_v<std::remove_const_t<DataT>, vec<half, 4>>;
        sample(*st_, xyz, fp, c);
        return to_data<DataT>(c);
    }
    std::shared_ptr<state> st_;
    property_list props_;
};
inline void check_no_init(const property_list& props, bool read_only) {
    if (read_only && props.template paras_has_property<property::no_init>())
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "property::no_init is not allowed on a read-only image accessor");
}
}
}

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget = image_target::device>
class unsampled_image_accessor : public detail::paras_img::acc_base<DataT, Dimensions> {
    static_assert(AccessMode == access_mode::read || AccessMode == access_mode::write,
                  "unsampled_image_accessor supports access_mode::read and access_mode::write");
    using base = detail::paras_img::acc_base<DataT, Dimensions>;

public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type&;
    using const_reference = const DataT&;

    template <typename AllocatorT>
    unsampled_image_accessor(unsampled_image<Dimensions, AllocatorT>& imageRef,
                             handler& commandGroupHandlerRef, const property_list& propList = {})
        : base(imageRef.paras_state(), propList) {
        detail::paras_img::check_no_init(propList, AccessMode == access_mode::read);
        if (AccessTarget == image_target::device &&
            !commandGroupHandlerRef.paras_device_has(aspect::image))
            throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                                  "the device does not have aspect::image");
    }

    template <typename OtherT,
              typename = std::enable_if_t<AccessMode == access_mode::read &&
                                          !std::is_same_v<OtherT, DataT> &&
                                          std::is_same_v<std::remove_const_t<OtherT>,
                                                         std::remove_const_t<DataT>>>>
    unsampled_image_accessor(
        const unsampled_image_accessor<OtherT, Dimensions, AccessMode, AccessTarget>& other)
        : base(other.paras_state(), {}) {}

    template <typename CoordT, access_mode M = AccessMode,
              typename = std::enable_if_t<M == access_mode::read &&
                                          detail::paras_img::int_coord_ok<Dimensions, CoordT>>>
    DataT read(const CoordT& coords) const noexcept {
        return this->do_read_int(coords);
    }
    template <typename CoordT, access_mode M = AccessMode,
              typename = std::enable_if_t<M == access_mode::write &&
                                          detail::paras_img::int_coord_ok<Dimensions, CoordT>>>
    void write(const CoordT& coords, const DataT& color) const {
        this->do_write(coords, color);
    }

    friend bool operator==(const unsampled_image_accessor& a, const unsampled_image_accessor& b) {
        return a.st_ == b.st_;
    }
    friend bool operator!=(const unsampled_image_accessor& a, const unsampled_image_accessor& b) {
        return !(a == b);
    }
};

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write)>
class host_unsampled_image_accessor : public detail::paras_img::acc_base<DataT, Dimensions> {
    static_assert(AccessMode == access_mode::read || AccessMode == access_mode::write ||
                      AccessMode == access_mode::read_write,
                  "host_unsampled_image_accessor supports read, write and read_write");
    using base = detail::paras_img::acc_base<DataT, Dimensions>;

public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type&;
    using const_reference = const DataT&;

    template <typename AllocatorT>
    host_unsampled_image_accessor(unsampled_image<Dimensions, AllocatorT>& imageRef,
                                  const property_list& propList = {})
        : base(imageRef.paras_state(), propList) {
        detail::paras_img::check_no_init(propList, AccessMode == access_mode::read);
        paras_drain_host_tasks();
    }

    template <typename OtherT,
              typename = std::enable_if_t<AccessMode == access_mode::read &&
                                          !std::is_same_v<OtherT, DataT> &&
                                          std::is_same_v<std::remove_const_t<OtherT>,
                                                         std::remove_const_t<DataT>>>>
    host_unsampled_image_accessor(
        const host_unsampled_image_accessor<OtherT, Dimensions, AccessMode>& other)
        : base(other.paras_state(), {}) {}

    template <typename CoordT, access_mode M = AccessMode,
              typename = std::enable_if_t<(M == access_mode::read || M == access_mode::read_write) &&
                                          detail::paras_img::int_coord_ok<Dimensions, CoordT>>>
    DataT read(const CoordT& coords) const noexcept {
        return this->do_read_int(coords);
    }
    template <typename CoordT, access_mode M = AccessMode,
              typename = std::enable_if_t<(M == access_mode::write || M == access_mode::read_write) &&
                                          detail::paras_img::int_coord_ok<Dimensions, CoordT>>>
    void write(const CoordT& coords, const DataT& color) const {
        this->do_write(coords, color);
    }

    friend bool operator==(const host_unsampled_image_accessor& a,
                           const host_unsampled_image_accessor& b) {
        return a.st_ == b.st_;
    }
    friend bool operator!=(const host_unsampled_image_accessor& a,
                           const host_unsampled_image_accessor& b) {
        return !(a == b);
    }
};

template <typename DataT, int Dimensions, image_target AccessTarget = image_target::device>
class sampled_image_accessor : public detail::paras_img::acc_base<DataT, Dimensions> {
    using base = detail::paras_img::acc_base<DataT, Dimensions>;

public:
    using value_type = const DataT;
    using reference = const DataT&;
    using const_reference = const DataT&;

    template <typename AllocatorT>
    sampled_image_accessor(sampled_image<Dimensions, AllocatorT>& imageRef,
                           handler& commandGroupHandlerRef, const property_list& propList = {})
        : base(imageRef.paras_state(), propList) {
        if (AccessTarget == image_target::device &&
            !commandGroupHandlerRef.paras_device_has(aspect::image))
            throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                                  "the device does not have aspect::image");
    }
    template <typename OtherT, typename = std::enable_if_t<!std::is_same_v<OtherT, DataT> &&
                                                           std::is_same_v<std::remove_const_t<OtherT>,
                                                                          std::remove_const_t<DataT>>>>
    sampled_image_accessor(const sampled_image_accessor<OtherT, Dimensions, AccessTarget>& other)
        : base(other.paras_state(), {}) {}

    template <typename CoordT,
              typename = std::enable_if_t<detail::paras_img::float_coord_ok<Dimensions, CoordT>>>
    DataT read(const CoordT& coords) const noexcept {
        return this->do_sample(coords);
    }

    friend bool operator==(const sampled_image_accessor& a, const sampled_image_accessor& b) {
        return a.st_ == b.st_;
    }
    friend bool operator!=(const sampled_image_accessor& a, const sampled_image_accessor& b) {
        return !(a == b);
    }
};

template <typename DataT, int Dimensions>
class host_sampled_image_accessor : public detail::paras_img::acc_base<DataT, Dimensions> {
    using base = detail::paras_img::acc_base<DataT, Dimensions>;

public:
    using value_type = const DataT;
    using reference = const DataT&;
    using const_reference = const DataT&;

    template <typename AllocatorT>
    host_sampled_image_accessor(sampled_image<Dimensions, AllocatorT>& imageRef,
                                const property_list& propList = {})
        : base(imageRef.paras_state(), propList) {
        paras_drain_host_tasks();
    }
    template <typename OtherT, typename = std::enable_if_t<!std::is_same_v<OtherT, DataT> &&
                                                           std::is_same_v<std::remove_const_t<OtherT>,
                                                                          std::remove_const_t<DataT>>>>
    host_sampled_image_accessor(const host_sampled_image_accessor<OtherT, Dimensions>& other)
        : base(other.paras_state(), {}) {}

    template <typename CoordT,
              typename = std::enable_if_t<detail::paras_img::float_coord_ok<Dimensions, CoordT>>>
    DataT read(const CoordT& coords) const noexcept {
        return this->do_sample(coords);
    }

    friend bool operator==(const host_sampled_image_accessor& a,
                           const host_sampled_image_accessor& b) {
        return a.st_ == b.st_;
    }
    friend bool operator!=(const host_sampled_image_accessor& a,
                           const host_sampled_image_accessor& b) {
        return !(a == b);
    }
};

template <int D, typename A>
struct is_property_of<property::image::use_host_ptr, unsampled_image<D, A>> : std::true_type {};
template <int D, typename A>
struct is_property_of<property::image::use_mutex, unsampled_image<D, A>> : std::true_type {};
template <int D, typename A>
struct is_property_of<property::image::context_bound, unsampled_image<D, A>> : std::true_type {};
template <int D, typename A>
struct is_property_of<property::image::use_host_ptr, sampled_image<D, A>> : std::true_type {};
template <int D, typename A>
struct is_property_of<property::image::use_mutex, sampled_image<D, A>> : std::true_type {};
template <int D, typename A>
struct is_property_of<property::image::context_bound, sampled_image<D, A>> : std::true_type {};
template <typename T, int D, access_mode M, image_target Tg>
struct is_property_of<property::no_init, unsampled_image_accessor<T, D, M, Tg>> : std::true_type {};
template <typename T, int D, access_mode M>
struct is_property_of<property::no_init, host_unsampled_image_accessor<T, D, M>> : std::true_type {};

}

template <int D, typename A>
struct std::hash<sycl::unsampled_image<D, A>> {
    std::size_t operator()(const sycl::unsampled_image<D, A>& i) const noexcept {
        return std::hash<const void*>{}(i.paras_state().get());
    }
};
template <int D, typename A>
struct std::hash<sycl::sampled_image<D, A>> {
    std::size_t operator()(const sycl::sampled_image<D, A>& i) const noexcept {
        return std::hash<const void*>{}(i.paras_state().get());
    }
};
template <typename T, int D, sycl::access_mode M, sycl::image_target Tg>
struct std::hash<sycl::unsampled_image_accessor<T, D, M, Tg>> {
    std::size_t operator()(const sycl::unsampled_image_accessor<T, D, M, Tg>& a) const noexcept {
        return std::hash<const void*>{}(a.paras_state().get());
    }
};
template <typename T, int D, sycl::access_mode M>
struct std::hash<sycl::host_unsampled_image_accessor<T, D, M>> {
    std::size_t operator()(const sycl::host_unsampled_image_accessor<T, D, M>& a) const noexcept {
        return std::hash<const void*>{}(a.paras_state().get());
    }
};
template <typename T, int D, sycl::image_target Tg>
struct std::hash<sycl::sampled_image_accessor<T, D, Tg>> {
    std::size_t operator()(const sycl::sampled_image_accessor<T, D, Tg>& a) const noexcept {
        return std::hash<const void*>{}(a.paras_state().get());
    }
};
template <typename T, int D>
struct std::hash<sycl::host_sampled_image_accessor<T, D>> {
    std::size_t operator()(const sycl::host_sampled_image_accessor<T, D>& a) const noexcept {
        return std::hash<const void*>{}(a.paras_state().get());
    }
};

#endif

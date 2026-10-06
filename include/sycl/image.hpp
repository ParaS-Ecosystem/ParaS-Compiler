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

#ifndef __PARAS_IMAGE_HPP__
#define __PARAS_IMAGE_HPP__

#include "range.hpp"
#include "property_list.hpp"
#include "access.hpp"
#include "accessor.hpp"
#include "handler.hpp"
#include <cstddef>
#include <cstring>

namespace sycl {

enum class image_channel_order {
    a,
    r,
    rx,
    rg,
    rgx,
    ra,
    rgb,
    rgbx,
    rgba,
    argb,
    bgra,
    intensity,
    luminance,
    abgr
};

enum class image_channel_type {
    snorm_int8,
    snorm_int16,
    unorm_int8,
    unorm_int16,
    unorm_short_565,
    unorm_short_555,
    unorm_int_101010,
    signed_int8,
    signed_int16,
    signed_int32,
    unsigned_int8,
    unsigned_int16,
    unsigned_int32,
    fp16,
    fp32
};

inline std::size_t paras_image_channel_type_size(image_channel_type type) {
    switch (type) {
    case image_channel_type::snorm_int8:
    case image_channel_type::unorm_int8:
    case image_channel_type::signed_int8:
    case image_channel_type::unsigned_int8:
        return 1;
    case image_channel_type::snorm_int16:
    case image_channel_type::unorm_int16:
    case image_channel_type::unorm_short_565:
    case image_channel_type::unorm_short_555:
    case image_channel_type::signed_int16:
    case image_channel_type::unsigned_int16:
    case image_channel_type::fp16:
        return 2;
    default:
        return 4;
    }
}

template <int Dimensions = 1, typename AllocatorT = std::allocator<std::byte>>
class image {
    unsigned char* data_;
    range<Dimensions> range_;
    image_channel_order order_;
    image_channel_type type_;
    bool owns_data_;

public:
    image(image_channel_order order, image_channel_type type, const range<Dimensions>& rangeRef,
          const property_list& = {})
        : range_(rangeRef), order_(order), type_(type), owns_data_(true) {
        data_ = new unsigned char[byte_size()]();
    }

    image(void* hostPointer, image_channel_order order, image_channel_type type,
          const range<Dimensions>& rangeRef, const property_list& = {})
        : data_(static_cast<unsigned char*>(hostPointer)), range_(rangeRef), order_(order),
          type_(type), owns_data_(false) {}

    ~image() {
        if (owns_data_)
            delete[] data_;
    }

    image(const image&) = delete;
    image& operator=(const image&) = delete;

    range<Dimensions> get_range() const { return range_; }

    std::size_t size() const noexcept {
        std::size_t n = 1;
        for (int d = 0; d < Dimensions; ++d)
            n *= range_[d];
        return n;
    }

    std::size_t byte_size() const noexcept { return size() * paras_image_channel_type_size(type_); }

    image_channel_order get_channel_order() const { return order_; }
    image_channel_type get_channel_type() const { return type_; }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    template <typename DataT, access::mode AccessMode = access::mode::read_write>
    accessor<DataT, Dimensions, AccessMode, target::image> get_access(handler&) {
        return accessor<DataT, Dimensions, AccessMode, target::image>(
            reinterpret_cast<DataT*>(data_), range_, AccessMode);
    }

    template <typename DataT, access::mode AccessMode = access::mode::read_write>
    accessor<DataT, Dimensions, AccessMode, target::host_image> get_access() {
        return accessor<DataT, Dimensions, AccessMode, target::host_image>(
            reinterpret_cast<DataT*>(data_), range_, AccessMode);
    }
#pragma clang diagnostic pop
};

} // namespace sycl

#endif

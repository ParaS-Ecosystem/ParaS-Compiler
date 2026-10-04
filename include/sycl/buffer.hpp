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

#ifndef __PARAS_BUFFER_HPP__
#define __PARAS_BUFFER_HPP__

#include <memory>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <functional>

#include "range.hpp"
#include "event.hpp"
#include "property_list.hpp"
#include "access.hpp"
#include "handler.hpp"
#include "accessor.hpp"

namespace sycl {

template <typename T>
class buffer_allocator : public std::allocator<T> {
public:
    buffer_allocator() = default;
    template <typename U>
    buffer_allocator(const buffer_allocator<U>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = buffer_allocator<U>;
    };
};

template <typename T, typename U>
bool operator==(const buffer_allocator<T>&, const buffer_allocator<U>&) noexcept {
    return true;
}
template <typename T, typename U>
bool operator!=(const buffer_allocator<T>&, const buffer_allocator<U>&) noexcept {
    return false;
}

template <
    typename T,
    int Dimensions = 1,
    typename AllocatorT = buffer_allocator<T>>
class buffer
{
    T* data_;
    range<Dimensions> r_;
    property_list props_;
    bool is_sub_buffer_ = false;

    template <typename>
    friend struct std::hash;

public:

    ~buffer() { ::sycl::detail::wait_all_host_tasks(); }
    buffer(const buffer &) = default;
    buffer(buffer &&) = default;
    buffer &operator=(const buffer &) = default;
    buffer &operator=(buffer &&) = default;

    using value_type      = T;
    using reference       = T&;
    using const_reference = const T&;
    using allocator_type  = AllocatorT;

    buffer(const range<Dimensions>& rr,
           const property_list& propList = {})
        : r_(rr), props_(propList)
    {

        size_t N = rr.size();
        data_ = static_cast<T*>(::operator new[](N * sizeof(T)));
    }

    buffer(T* hostPtr,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : data_(hostPtr), r_(rr), props_(propList) {}

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    buffer(const T* hostPtr,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : r_(rr), props_(propList)
    {

        size_t N = rr.size();
        data_ = static_cast<T*>(::operator new[](N * sizeof(T)));
        for (size_t i = 0; i < N; ++i) ::new (data_ + i) T(hostPtr[i]);
    }

    buffer(const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(rr, propList) {}

    buffer(T* hostPtr, const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostPtr, rr, propList) {}

    template <typename U = T,
              typename = std::enable_if_t<!std::is_const_v<U>>>
    buffer(const T* hostPtr, const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostPtr, rr, propList) {}

    template <typename InputIterator>
    buffer(InputIterator first, InputIterator last,
           const property_list& propList = {})
        : r_(range<Dimensions>(static_cast<std::size_t>(std::distance(first, last)))), props_(propList)
    {

        std::size_t n = static_cast<std::size_t>(std::distance(first, last));
        data_ = static_cast<T*>(::operator new[](n * sizeof(T)));
        std::size_t i = 0;
        for (auto it = first; it != last; ++it, ++i) ::new (data_ + i) T(*it);
    }

    template <typename InputIterator>
    buffer(InputIterator first, InputIterator last, AllocatorT,
           const property_list& propList = {})
        : buffer(first, last, propList) {}

    template <typename Container,
              typename = decltype(std::declval<Container&>().data()),
              typename = decltype(std::declval<Container&>().size()),
              int D = Dimensions,
              typename = std::enable_if_t<D == 1>>
    buffer(Container& container, AllocatorT,
           const property_list& propList = {})
        : buffer(container.data(), range<1>(container.size()), propList) {}

    template <typename Container,
              typename = decltype(std::declval<Container&>().data()),
              typename = decltype(std::declval<Container&>().size()),
              int D = Dimensions,
              typename = std::enable_if_t<D == 1>>
    buffer(Container& container,
           const property_list& propList = {})
        : buffer(container.data(), range<1>(container.size()), propList) {}

    buffer(const std::shared_ptr<T>& hostData,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : data_(hostData.get()), r_(rr), props_(propList) {}

    buffer(const std::shared_ptr<T>& hostData,
           const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostData, rr, propList) {}

    buffer(const std::shared_ptr<T[]>& hostData,
           const range<Dimensions>& rr,
           const property_list& propList = {})
        : data_(hostData.get()), r_(rr), props_(propList) {}

    buffer(const std::shared_ptr<T[]>& hostData,
           const range<Dimensions>& rr, AllocatorT,
           const property_list& propList = {})
        : buffer(hostData, rr, propList) {}

    template <typename PropT>
    bool has_property() const noexcept {
        return props_.template has_property<PropT>();
    }

    template <typename PropT>
    PropT get_property() const {
        return props_.template get_property<PropT>();
    }

    template <typename T2>
    void set_final_data(T2) {}
    void set_final_data(std::nullptr_t) {}

    void set_write_back(bool = true) {}

    bool is_sub_buffer() const noexcept { return is_sub_buffer_; }
    AllocatorT get_allocator() const { return AllocatorT{}; }

    buffer(buffer& b, id<Dimensions> baseIndex, range<Dimensions> subRange)
        : r_(subRange), props_(b.props_), is_sub_buffer_(true)
    {
        std::size_t flat_offset = baseIndex[0];
        if constexpr (Dimensions == 2) {
            flat_offset = baseIndex[0] * b.r_[1] + baseIndex[1];
        } else if constexpr (Dimensions == 3) {
            flat_offset = (baseIndex[0] * b.r_[1] + baseIndex[1]) * b.r_[2] + baseIndex[2];
        }
        data_ = b.data_ + flat_offset;
    }

    friend bool operator==(const buffer &lhs, const buffer &rhs) noexcept {
        return lhs.data_ == rhs.data_;
    }
    friend bool operator!=(const buffer &lhs, const buffer &rhs) noexcept {
        return !(lhs == rhs);
    }

    template <typename ReinterpretT, int ReinterpretDim = Dimensions>
    buffer<ReinterpretT, ReinterpretDim> reinterpret(range<ReinterpretDim> reinterpretRange) const
    {
        return buffer<ReinterpretT, ReinterpretDim>(
            reinterpret_cast<ReinterpretT*>(data_), reinterpretRange, props_);
    }

    template <typename ReinterpretT, int ReinterpretDim = Dimensions>
    buffer<ReinterpretT, ReinterpretDim> reinterpret() const
    {
        std::size_t total_bytes = r_.size() * sizeof(T);
        std::size_t new_count = total_bytes / sizeof(ReinterpretT);
        if constexpr (ReinterpretDim == 1) {
            return reinterpret<ReinterpretT, 1>(range<1>(new_count));
        } else {
            range<ReinterpretDim> new_range;
            for (int i = 0; i < ReinterpretDim; ++i) new_range[i] = r_[i];
            new_range[ReinterpretDim - 1] =
                (r_[Dimensions - 1] * sizeof(T)) / sizeof(ReinterpretT);
            return reinterpret<ReinterpretT, ReinterpretDim>(new_range);
        }
    }

    template <
        access::mode Mode = access::mode::read_write,
        target AccessTarget = target::device>
    accessor<T, Dimensions, Mode, AccessTarget>
    get_access(handler& cgh)
    {
        return accessor<T, Dimensions, Mode, AccessTarget>(
            data_, r_, Mode
        );
    }

    template <
        access::mode Mode = access::mode::read_write,
        target AccessTarget = target::device>
    accessor<T, Dimensions, Mode, AccessTarget>
    get_access(handler& cgh, range<Dimensions> accessRange, id<Dimensions> accessOffset)
    {
        (void)cgh;
        std::size_t flat_offset = accessOffset[0];
        if constexpr (Dimensions >= 2) {
            flat_offset = accessOffset[0] * r_[1] + accessOffset[1];
        }
        return accessor<T, Dimensions, Mode, AccessTarget>(
            data_ + flat_offset, accessRange, Mode
        );
    }

    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer>
    get_access()
    {
        return accessor<T, Dimensions, Mode, target::host_buffer>(
            data_, r_, Mode
        );
    }

    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer>
    get_access(range<Dimensions> accessRange, id<Dimensions> accessOffset)
    {
        std::size_t flat_offset = accessOffset[0];
        if constexpr (Dimensions >= 2) {
            flat_offset = accessOffset[0] * r_[1] + accessOffset[1];
        }
        return accessor<T, Dimensions, Mode, target::host_buffer>(
            data_ + flat_offset, accessRange, Mode
        );
    }

    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer>
    get_access(range<Dimensions> accessRange)
    {
        return get_access<Mode>(accessRange, id<Dimensions>());
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::device>
    get_access(handler& cgh, TagT)
    {
        (void)cgh;
        return accessor<T, Dimensions, TagT::value, target::device>(
            data_, r_, TagT::value
        );
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::device>
    get_access(handler& cgh, TagT, property::no_init)
    {
        return get_access(cgh, TagT{});
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::device>
    get_access(handler& cgh, range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT)
    {
        return get_access<TagT::value, target::device>(cgh, accessRange, accessOffset);
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::host_buffer>
    get_access(TagT)
    {
        return get_access<TagT::value>();
    }

    template <typename TagT>
    accessor<T, Dimensions, TagT::value, target::host_buffer>
    get_access(range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT)
    {
        return get_access<TagT::value>(accessRange, accessOffset);
    }

    accessor<T, Dimensions, access::mode::read_write, target::host_buffer>
    get_host_access()
    {
        return accessor<T, Dimensions, access::mode::read_write, target::host_buffer>(
            data_, r_, access::mode::read_write
        );
    }

    T* data() const { return data_; }

    range<Dimensions> get_range() const noexcept { return r_; }

    std::size_t size() const noexcept { return r_.size(); }

    std::size_t byte_size() const noexcept { return r_.size() * sizeof(T); }

    [[deprecated("use size()")]] std::size_t get_count() const { return size(); }
    [[deprecated("use byte_size()")]] std::size_t get_size() const { return byte_size(); }

};

template <typename InputIterator,
          typename T = typename std::iterator_traits<InputIterator>::value_type>
buffer(InputIterator, InputIterator, const property_list& = {})
    -> buffer<T, 1, buffer_allocator<T>>;

template <typename InputIterator, typename AllocatorT,
          typename T = typename std::iterator_traits<InputIterator>::value_type>
buffer(InputIterator, InputIterator, AllocatorT, const property_list& = {})
    -> buffer<T, 1, AllocatorT>;

template <typename Container,
          typename T = typename Container::value_type>
buffer(Container&, const property_list& = {})
    -> buffer<T, 1, buffer_allocator<T>>;

template <typename Container, typename AllocatorT,
          typename T = typename Container::value_type>
buffer(Container&, AllocatorT, const property_list& = {})
    -> buffer<T, 1, AllocatorT>;

}

template <typename T, int Dimensions, typename AllocatorT>
struct std::hash<sycl::buffer<T, Dimensions, AllocatorT>> {
    std::size_t operator()(const sycl::buffer<T, Dimensions, AllocatorT>& b) const noexcept {
        return std::hash<const void*>{}(static_cast<const void*>(b.data_));
    }
};

#endif

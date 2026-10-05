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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include "range.hpp"
#include "id.hpp"
#include "property_list.hpp"
#include "access.hpp"
#include "handler.hpp"
#include "accessor.hpp"

namespace sycl {

template <typename T, int Dimensions = 1, typename AllocatorT = std::allocator<T>>
class buffer {
    struct state {
        T* data = nullptr;
        range<Dimensions> r;
        std::shared_ptr<T> keep_alive;                         // owned storage or user's shared_ptr
        std::function<void(const T*, std::size_t)> final_data; // write-back target
        bool write_back = true;

        ~state() {
            if (write_back && final_data)
                final_data(data, r.size());
        }
    };
    std::shared_ptr<state> st_;

    template <typename Init>
    static std::shared_ptr<T> allocate_owned(std::size_t n, Init init) {
        T* p = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
        try {
            init(p);
        } catch (...) {
            ::operator delete(p, std::align_val_t(alignof(T)));
            throw;
        }
        return std::shared_ptr<T>(p, [n](T* q) {
            std::destroy_n(q, n);
            ::operator delete(q, std::align_val_t(alignof(T)));
        });
    }

    void init_owned(const range<Dimensions>& r) {
        static_assert(std::is_default_constructible_v<T>,
                      "buffer(range) needs a default-constructible element type");
        const std::size_t n = r.size();
        init_owned_with(r, [n](T* p) { std::uninitialized_value_construct_n(p, n); });
    }

    template <typename InputIterator>
    void init_owned_copy(const range<Dimensions>& r, InputIterator first) {
        const std::size_t n = r.size();
        init_owned_with(r, [n, first](T* p) { std::uninitialized_copy_n(first, n, p); });
    }

    template <typename Init>
    void init_owned_with(const range<Dimensions>& r, Init init) {
        st_ = std::make_shared<state>();
        st_->r = r;
        st_->keep_alive = allocate_owned(r.size(), init);
        st_->data = st_->keep_alive.get();
    }

public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using allocator_type = AllocatorT;

    buffer(const range<Dimensions>& r, const property_list& = {}) { init_owned(r); }
    buffer(const range<Dimensions>& r, AllocatorT, const property_list& props = {})
        : buffer(r, props) {}

    buffer(T* hostData, const range<Dimensions>& r, const property_list& = {}) {
        st_ = std::make_shared<state>();
        st_->r = r;
        st_->data = hostData;
    }
    buffer(T* hostData, const range<Dimensions>& r, AllocatorT, const property_list& props = {})
        : buffer(hostData, r, props) {}

    buffer(const T* hostData, const range<Dimensions>& r, const property_list& = {}) {
        init_owned_copy(r, hostData);
    }
    buffer(const T* hostData, const range<Dimensions>& r, AllocatorT,
           const property_list& props = {})
        : buffer(hostData, r, props) {}

    buffer(const std::shared_ptr<T>& hostData, const range<Dimensions>& r,
           const property_list& = {}) {
        st_ = std::make_shared<state>();
        st_->r = r;
        st_->keep_alive = hostData;
        st_->data = hostData.get();
    }
    buffer(const std::shared_ptr<T>& hostData, const range<Dimensions>& r, AllocatorT,
           const property_list& props = {})
        : buffer(hostData, r, props) {}
    buffer(const std::shared_ptr<T[]>& hostData, const range<Dimensions>& r,
           const property_list& props = {})
        : buffer(std::shared_ptr<T>(hostData, hostData.get()), r, props) {}

    template <typename InputIterator,
              typename = std::enable_if_t<!std::is_integral_v<InputIterator>>>
    buffer(InputIterator first, InputIterator last, const property_list& = {}) {
        static_assert(Dimensions == 1, "iterator buffer constructor requires Dimensions == 1");
        const std::size_t n = static_cast<std::size_t>(std::distance(first, last));
        init_owned_copy(range<Dimensions>(n), first);
    }
    template <typename InputIterator,
              typename = std::enable_if_t<!std::is_integral_v<InputIterator>>>
    buffer(InputIterator first, InputIterator last, AllocatorT, const property_list& props = {})
        : buffer(first, last, props) {}

    buffer(const buffer&) = default;
    buffer(buffer&&) noexcept = default;
    buffer& operator=(const buffer&) = default;
    buffer& operator=(buffer&&) noexcept = default;

    bool operator==(const buffer& rhs) const noexcept { return st_ == rhs.st_; }
    bool operator!=(const buffer& rhs) const noexcept { return st_ != rhs.st_; }

    template <access::mode Mode = access::mode::read_write, target AccessTarget = target::device>
    accessor<T, Dimensions, Mode, AccessTarget> get_access(handler&) {
        return accessor<T, Dimensions, Mode, AccessTarget>(st_->data, st_->r, Mode);
    }
    template <access::mode Mode = access::mode::read_write, target AccessTarget = target::device>
    accessor<T, Dimensions, Mode, AccessTarget> get_access(handler&, range<Dimensions> accessRange,
                                                           id<Dimensions> accessOffset = {}) {
        return accessor<T, Dimensions, Mode, AccessTarget>(st_->data, st_->r, accessRange,
                                                           accessOffset);
    }
    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer> get_access() {
        return accessor<T, Dimensions, Mode, target::host_buffer>(st_->data, st_->r, Mode);
    }
    template <access::mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer>
    get_access(range<Dimensions> accessRange, id<Dimensions> accessOffset = {}) {
        return accessor<T, Dimensions, Mode, target::host_buffer>(st_->data, st_->r, accessRange,
                                                                  accessOffset);
    }
    accessor<T, Dimensions, access::mode::read_write, target::host_buffer> get_host_access() {
        return accessor<T, Dimensions, access::mode::read_write, target::host_buffer>(
            st_->data, st_->r, access::mode::read_write);
    }
    template <typename TagT>
    auto get_host_access(TagT) {
        return accessor<T, Dimensions, TagT::value, target::host_buffer>(st_->data, st_->r,
                                                                         TagT::value);
    }

    void set_final_data(std::nullptr_t) { st_->final_data = nullptr; }
    void set_final_data(T* finalData) {
        if (finalData == st_->data) {
            st_->final_data = nullptr;
            return;
        }
        st_->final_data = [finalData](const T* src, std::size_t n) {
            std::copy(src, src + n, finalData);
        };
    }
    void set_final_data(std::shared_ptr<T> finalData) {
        if (finalData.get() == st_->data) {
            st_->final_data = nullptr;
            return;
        }
        st_->final_data = [finalData](const T* src, std::size_t n) {
            std::copy(src, src + n, finalData.get());
        };
    }
    template <typename Destination,
              typename = std::enable_if_t<!std::is_pointer_v<Destination> &&
                                          !std::is_same_v<Destination, std::nullptr_t>>>
    void set_final_data(Destination finalData) {
        st_->final_data = [finalData](const T* src, std::size_t n) mutable {
            std::copy(src, src + n, finalData);
        };
    }
    void set_write_back(bool flag = true) { st_->write_back = flag; }

    range<Dimensions> get_range() const noexcept { return st_->r; }
    std::size_t size() const noexcept { return st_->r.size(); }
    std::size_t byte_size() const noexcept { return size() * sizeof(T); }
    std::size_t get_count() const noexcept { return size(); }
    std::size_t get_size() const noexcept { return byte_size(); }
    AllocatorT get_allocator() const { return AllocatorT(); }
    template <typename Property>
    bool has_property() const noexcept {
        return false;
    }
    T* data() const noexcept { return st_->data; }
};

template <class InputIterator, class AllocatorT>
buffer(InputIterator, InputIterator, AllocatorT, const property_list& = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1, AllocatorT>;
template <class InputIterator>
buffer(InputIterator, InputIterator, const property_list& = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;
template <class T, int Dimensions, class AllocatorT>
buffer(const T*, const range<Dimensions>&, AllocatorT, const property_list& = {})
    -> buffer<T, Dimensions, AllocatorT>;
template <class T, int Dimensions>
buffer(const T*, const range<Dimensions>&, const property_list& = {}) -> buffer<T, Dimensions>;

} // namespace sycl

namespace std {
template <typename T, int Dimensions, typename AllocatorT>
struct hash<sycl::buffer<T, Dimensions, AllocatorT>> {
    size_t operator()(const sycl::buffer<T, Dimensions, AllocatorT>& b) const noexcept {
        return std::hash<const void*>{}(static_cast<const void*>(b.data()));
    }
};
} // namespace std

#endif

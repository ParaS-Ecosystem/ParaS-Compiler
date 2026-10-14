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
namespace detail {
template <typename U>
struct paras_identity {
    using type = U;
};
}

template <typename T>
class buffer_allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    template <typename U>
    struct rebind {
        using other = buffer_allocator<U>;
    };
    buffer_allocator() noexcept = default;
    template <typename U>
    buffer_allocator(const buffer_allocator<U>&) noexcept {}
    T* allocate(std::size_t n) { return std::allocator<T>{}.allocate(n); }
    void deallocate(T* p, std::size_t n) { std::allocator<T>{}.deallocate(p, n); }
    template <typename U>
    bool operator==(const buffer_allocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const buffer_allocator<U>&) const noexcept { return false; }
};

template <typename T, int Dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
class buffer {
    template <typename, int, typename>
    friend class buffer;

    struct state {
        T* data = nullptr;
        range<Dimensions> r;
        std::shared_ptr<T> keep_alive;                         // owned storage or user's shared_ptr
        std::function<void(const T*, std::size_t)> final_data; // write-back target
        bool write_back = true;
        property_list props;
        bool sub_buffer = false;
        std::shared_ptr<const void> parent;
        AllocatorT alloc{};

        ~state() {
            if (write_back && final_data)
                final_data(data, r.size());
        }
    };
    std::shared_ptr<state> st_;

    template <typename Init>
    static std::shared_ptr<T> allocate_owned(AllocatorT alloc, std::size_t n, Init init) {
        using V = std::remove_const_t<T>;
        using VA = typename std::allocator_traits<AllocatorT>::template rebind_alloc<V>;
        using VT = std::allocator_traits<VA>;
        VA va(alloc);
        V* p = VT::allocate(va, n);
        try {
            init(const_cast<V*>(p));
        } catch (...) {
            VT::deallocate(va, p, n);
            throw;
        }
        return std::shared_ptr<T>(p, [va, n](T* q) mutable {
            V* v = const_cast<V*>(q);
            std::destroy_n(v, n);
            VT::deallocate(va, v, n);
        });
    }

    void init_owned(const range<Dimensions>& r, AllocatorT alloc = AllocatorT{}) {
        static_assert(std::is_default_constructible_v<T>,
                      "buffer(range) needs a default-constructible element type");
        const std::size_t n = r.size();
        init_owned_with(
            r, [n](std::remove_const_t<T>* p) { std::uninitialized_value_construct_n(p, n); },
            alloc);
    }

    template <typename InputIterator>
    void init_owned_copy(const range<Dimensions>& r, InputIterator first,
                         AllocatorT alloc = AllocatorT{}) {
        const std::size_t n = r.size();
        init_owned_with(
            r, [n, first](std::remove_const_t<T>* p) { std::uninitialized_copy_n(first, n, p); },
            alloc);
    }

    template <typename Init>
    void init_owned_with(const range<Dimensions>& r, Init init, AllocatorT alloc = AllocatorT{}) {
        st_ = std::make_shared<state>();
        st_->r = r;
        st_->alloc = alloc;
        st_->keep_alive = allocate_owned(alloc, r.size(), init);
        st_->data = st_->keep_alive.get();
    }

    template <typename C>
    struct paras_is_buffer : std::false_type {};
    template <typename U, int D, typename A>
    struct paras_is_buffer<buffer<U, D, A>> : std::true_type {};

    template <typename D>
    struct paras_is_weak_ptr : std::false_type {};
    template <typename U>
    struct paras_is_weak_ptr<std::weak_ptr<U>> : std::true_type {};
    template <typename D>
    struct paras_is_shared_array : std::false_type {};
    template <typename U>
    struct paras_is_shared_array<std::shared_ptr<U[]>> : std::true_type {};

public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using allocator_type = AllocatorT;

    buffer(const range<Dimensions>& r, const property_list& props = {}) {
        init_owned(r);
        st_->props = props;
    }
    buffer(const range<Dimensions>& r, AllocatorT allocator, const property_list& props = {}) {
        init_owned(r, allocator);
        st_->props = props;
    }

    buffer(T* hostData, const range<Dimensions>& r, const property_list& props = {}) {
        st_ = std::make_shared<state>();
        st_->r = r;
        st_->data = hostData;
        st_->props = props;
    }
    buffer(T* hostData, const range<Dimensions>& r, AllocatorT allocator,
           const property_list& props = {})
        : buffer(hostData, r, props) {
        st_->alloc = allocator;
    }

    template <typename U = T, typename = std::enable_if_t<!std::is_const_v<U>>>
    buffer(const typename detail::paras_identity<U>::type* hostData, const range<Dimensions>& r,
           const property_list& props = {}) {
        if (props.template paras_has_property<property::buffer::use_host_ptr>()) {
            st_ = std::make_shared<state>();
            st_->r = r;
            st_->data = const_cast<T*>(hostData);
        } else {
            init_owned_copy(r, hostData);
        }
        st_->props = props;
    }
    template <typename U = T, typename = std::enable_if_t<!std::is_const_v<U>>>
    buffer(const typename detail::paras_identity<U>::type* hostData, const range<Dimensions>& r,
           AllocatorT allocator, const property_list& props = {}) {
        if (props.template paras_has_property<property::buffer::use_host_ptr>()) {
            st_ = std::make_shared<state>();
            st_->r = r;
            st_->data = const_cast<T*>(hostData);
            st_->alloc = allocator;
        } else {
            init_owned_copy(r, hostData, allocator);
        }
        st_->props = props;
    }

    buffer(const std::shared_ptr<T>& hostData, const range<Dimensions>& r,
           const property_list& props = {}) {
        st_ = std::make_shared<state>();
        st_->r = r;
        st_->keep_alive = hostData;
        st_->data = hostData.get();
        st_->props = props;
    }
    buffer(const std::shared_ptr<T>& hostData, const range<Dimensions>& r, AllocatorT allocator,
           const property_list& props = {})
        : buffer(hostData, r, props) {
        st_->alloc = allocator;
    }
    buffer(const std::shared_ptr<T[]>& hostData, const range<Dimensions>& r, AllocatorT allocator,
           const property_list& props = {})
        : buffer(std::shared_ptr<T>(hostData, hostData.get()), r, allocator, props) {}
    buffer(const std::shared_ptr<T[]>& hostData, const range<Dimensions>& r,
           const property_list& props = {})
        : buffer(std::shared_ptr<T>(hostData, hostData.get()), r, props) {}

    template <typename Container, int D = Dimensions,
              typename = std::enable_if_t<
                  D == 1 && !paras_is_buffer<std::remove_cv_t<Container>>::value &&
                  std::is_convertible_v<decltype(std::data(std::declval<Container&>())), T*>>,
              typename = decltype(std::size(std::declval<Container&>()))>
    buffer(Container& container, const property_list& props = {})
        : buffer(std::data(container), range<1>(std::size(container)), props) {}
    template <typename Container, int D = Dimensions,
              typename = std::enable_if_t<
                  D == 1 && !paras_is_buffer<std::remove_cv_t<Container>>::value &&
                  std::is_convertible_v<decltype(std::data(std::declval<Container&>())), T*>>,
              typename = decltype(std::size(std::declval<Container&>()))>
    buffer(Container& container, AllocatorT allocator, const property_list& props = {})
        : buffer(std::data(container), range<1>(std::size(container)), allocator, props) {}

    template <typename InputIterator,
              typename = std::enable_if_t<!std::is_integral_v<InputIterator>>>
    buffer(InputIterator first, InputIterator last, const property_list& props = {}) {
        static_assert(Dimensions == 1, "iterator buffer constructor requires Dimensions == 1");
        const std::size_t n = static_cast<std::size_t>(std::distance(first, last));
        init_owned_copy(range<Dimensions>(n), first);
        st_->props = props;
    }

    buffer(buffer& b, const id<Dimensions>& baseIndex, const range<Dimensions>& subRange) {
        const range<Dimensions> pr = b.get_range();
        bool contiguous = true;
        int first_wide = Dimensions;
        for (int d = 0; d < Dimensions; ++d) {
            if (baseIndex[d] + subRange[d] > pr[d])
                throw sycl::exception(make_error_code(errc::invalid),
                                      "sub-buffer exceeds the parent buffer");
            if (first_wide == Dimensions && subRange[d] > 1)
                first_wide = d;
            else if (d > first_wide && subRange[d] != pr[d])
                contiguous = false;
        }
        if (!contiguous)
            throw sycl::exception(make_error_code(errc::invalid),
                                  "sub-buffer is not contiguous in the parent");
        if (b.is_sub_buffer())
            throw sycl::exception(make_error_code(errc::invalid),
                                  "cannot create a sub-buffer of a sub-buffer");
        std::size_t lin = 0;
        for (int d = 0; d < Dimensions; ++d)
            lin = lin * pr[d] + baseIndex[d];
        st_ = std::make_shared<state>();
        st_->r = subRange;
        st_->data = b.st_->data + lin;
        st_->props = b.st_->props;
        st_->sub_buffer = true;
        st_->parent = b.st_;
    }

    template <typename ReinterpretT, int ReinterpretDim>
    buffer<ReinterpretT, ReinterpretDim,
           typename std::allocator_traits<AllocatorT>::template rebind_alloc<ReinterpretT>>
    reinterpret(range<ReinterpretDim> reinterpretRange) const {
        using out_t = buffer<ReinterpretT, ReinterpretDim,
                             typename std::allocator_traits<AllocatorT>::template rebind_alloc<ReinterpretT>>;
        if (reinterpretRange.size() * sizeof(ReinterpretT) != byte_size())
            throw sycl::exception(make_error_code(errc::invalid),
                                  "reinterpret: total byte size differs");
        out_t out(reinterpret_cast<ReinterpretT*>(st_->data), reinterpretRange);
        out.st_->props = st_->props;
        out.st_->sub_buffer = st_->sub_buffer;
        out.st_->parent = st_;
        return out;
    }
    template <typename ReinterpretT, int ReinterpretDim = Dimensions,
              typename = std::enable_if_t<ReinterpretDim == 1 ||
                                          (ReinterpretDim == Dimensions &&
                                           sizeof(ReinterpretT) == sizeof(T))>>
    buffer<ReinterpretT, ReinterpretDim,
           typename std::allocator_traits<AllocatorT>::template rebind_alloc<ReinterpretT>>
    reinterpret() const {
        if constexpr (ReinterpretDim == 1)
            return reinterpret<ReinterpretT, 1>(range<1>(byte_size() / sizeof(ReinterpretT)));
        else
            return reinterpret<ReinterpretT, ReinterpretDim>(st_->r);
    }

    bool is_sub_buffer() const noexcept { return st_->sub_buffer; }
    template <typename InputIterator,
              typename = std::enable_if_t<!std::is_integral_v<InputIterator>>>
    buffer(InputIterator first, InputIterator last, AllocatorT allocator,
           const property_list& props = {}) {
        static_assert(Dimensions == 1, "iterator buffer constructor requires Dimensions == 1");
        const std::size_t n = static_cast<std::size_t>(std::distance(first, last));
        init_owned_copy(range<Dimensions>(n), first, allocator);
        st_->props = props;
    }

    buffer(const buffer&) = default;
    buffer(buffer&&) noexcept = default;

    ~buffer() { paras_drain_host_tasks(); }
    buffer& operator=(const buffer&) = default;
    buffer& operator=(buffer&&) noexcept = default;

    bool operator==(const buffer& rhs) const noexcept { return st_ == rhs.st_; }
    bool operator!=(const buffer& rhs) const noexcept { return st_ != rhs.st_; }

    template <access_mode Mode = access_mode::read_write, target Targ = target::device>
    accessor<T, Dimensions, Mode, Targ> get_access(handler& commandGroupHandler) {
        return accessor<T, Dimensions, Mode, Targ>(*this, commandGroupHandler);
    }
    template <access_mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer> get_access() {
        return accessor<T, Dimensions, Mode, target::host_buffer>(*this);
    }
    template <access_mode Mode = access_mode::read_write, target Targ = target::device>
    accessor<T, Dimensions, Mode, Targ> get_access(handler& commandGroupHandler,
                                                   range<Dimensions> accessRange,
                                                   id<Dimensions> accessOffset = {}) {
        return accessor<T, Dimensions, Mode, Targ>(*this, commandGroupHandler, accessRange,
                                                   accessOffset);
    }
    template <access_mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer> get_access(range<Dimensions> accessRange,
                                                                  id<Dimensions> accessOffset = {}) {
        return accessor<T, Dimensions, Mode, target::host_buffer>(*this, accessRange,
                                                                  accessOffset);
    }
    template <typename... Ts>
    auto get_access(Ts&&... args) {
        return accessor{*this, std::forward<Ts>(args)...};
    }
    template <typename... Ts>
    auto get_host_access(Ts&&... args) {
        return host_accessor{*this, std::forward<Ts>(args)...};
    }

    void set_final_data(std::nullptr_t) { st_->final_data = nullptr; }
    void set_final_data(T* finalData) {
        if (finalData == nullptr) {
            st_->final_data = nullptr;
            return;
        }
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
        if constexpr (paras_is_weak_ptr<Destination>::value) {
            st_->final_data = [finalData](const T* src, std::size_t n) {
                if (auto sp = finalData.lock())
                    std::copy(src, src + n, &sp.get()[0]);
            };
        } else if constexpr (paras_is_shared_array<Destination>::value) {
            st_->final_data = [finalData](const T* src, std::size_t n) {
                if (finalData)
                    std::copy(src, src + n, finalData.get());
            };
        } else {
            st_->final_data = [finalData](const T* src, std::size_t n) mutable {
                std::copy(src, src + n, finalData);
            };
        }
    }
    void set_write_back(bool flag = true) { st_->write_back = flag; }

    range<Dimensions> get_range() const noexcept { return st_->r; }
    std::size_t size() const noexcept { return st_->r.size(); }
    std::size_t byte_size() const noexcept { return size() * sizeof(T); }
    std::size_t get_count() const noexcept { return size(); }
    std::size_t get_size() const noexcept { return byte_size(); }
    AllocatorT get_allocator() const { return st_->alloc; }
    template <typename Property>
    bool has_property() const noexcept {
        return st_->props.template paras_has_property<Property>();
    }
    template <typename Property>
    Property get_property() const {
        return st_->props.template paras_get_property<Property>();
    }
    T* data() const noexcept { return st_->data; }
};

template <typename T, int D, typename A>
struct is_property_of<property::buffer::use_host_ptr, buffer<T, D, A>> : std::true_type {};
template <typename T, int D, typename A>
struct is_property_of<property::buffer::use_mutex, buffer<T, D, A>> : std::true_type {};
template <typename T, int D, typename A>
struct is_property_of<property::buffer::context_bound, buffer<T, D, A>> : std::true_type {};

template <class Container, class AllocatorT>
buffer(Container&, AllocatorT, const property_list& = {})
    -> buffer<typename Container::value_type, 1, AllocatorT>;
template <class Container>
buffer(Container&, const property_list& = {}) -> buffer<typename Container::value_type, 1>;

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

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

#ifndef __PARAS_ACCESSOR_HPP__
#define __PARAS_ACCESSOR_HPP__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <iterator>
#include <vector>
#include "range.hpp"
#include "id.hpp"
#include "access.hpp"
#include "property_list.hpp"
#include "handler.hpp"
#include "multi_ptr.hpp"
#include "atomic.hpp"
#include "global_ptr.hpp"
#include "local_ptr.hpp"
#include "sampler.hpp"
#include "exception.hpp"
#include "../utilities/internal_utils.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

namespace detail {
template <typename T, int Dims, int Level>
struct accessor_subscript {
    T* ptr;
    range<Dims> r;
    size_t lin;

    PARAS_KERNEL_HD
    decltype(auto) operator[](size_t i) const {
        const size_t next = lin * r[Level] + i;
        if constexpr (Level + 1 == Dims)
            return (ptr[next]);
        else
            return accessor_subscript<T, Dims, Level + 1>{ptr, r, next};
    }
};
} // namespace detail

template <typename T, int Dimensions, typename AllocatorT>
class buffer;

namespace detail {

template <typename T>
struct acc_tag : std::false_type {
    static constexpr bool host_accessor_ok = false;
};
#define PARAS_ACC_TAG(TAG, MODE, TARGET, HOST_OK)                                                  \
    template <>                                                                                    \
    struct acc_tag<TAG> : std::true_type {                                                         \
        static constexpr access_mode mode = access_mode::MODE;                                     \
        static constexpr target tgt = target::TARGET;                                              \
        static constexpr bool host_accessor_ok = HOST_OK;                                          \
    };
PARAS_ACC_TAG(read_write_tag_t, read_write, device, true)
PARAS_ACC_TAG(read_only_tag_t, read, device, true)
PARAS_ACC_TAG(write_only_tag_t, write, device, true)
PARAS_ACC_TAG(read_write_host_task_tag_t, read_write, host_task, false)
PARAS_ACC_TAG(read_only_host_task_tag_t, read, host_task, false)
PARAS_ACC_TAG(write_only_host_task_tag_t, write, host_task, false)
#undef PARAS_ACC_TAG
template <typename T>
inline constexpr bool is_acc_tag_v = acc_tag<std::decay_t<T>>::value;
template <typename T>
inline constexpr bool is_host_acc_tag_v =
    acc_tag<std::decay_t<T>>::value && acc_tag<std::decay_t<T>>::host_accessor_ok;

template <typename DataT>
inline constexpr access_mode default_acc_mode_v =
    std::is_const_v<DataT> ? access_mode::read : access_mode::read_write;

template <typename T, int Dims>
class acc_iterator {
public:
    using value_type = std::remove_const_t<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::random_access_iterator_tag;

    acc_iterator() = default;
    acc_iterator(T* base, range<Dims> buf, range<Dims> acc, id<Dims> off, difference_type pos)
        : base_(base), buf_(buf), acc_(acc), off_(off), pos_(pos) {}
    template <typename U = T, typename = std::enable_if_t<!std::is_const_v<U>>>
    operator acc_iterator<const T, Dims>() const {
        return acc_iterator<const T, Dims>(base_, buf_, acc_, off_, pos_);
    }

    reference operator*() const { return base_[address(pos_)]; }
    pointer operator->() const { return base_ + address(pos_); }
    reference operator[](difference_type n) const { return base_[address(pos_ + n)]; }

    acc_iterator& operator++() {
        ++pos_;
        return *this;
    }
    acc_iterator operator++(int) {
        acc_iterator t(*this);
        ++pos_;
        return t;
    }
    acc_iterator& operator--() {
        --pos_;
        return *this;
    }
    acc_iterator operator--(int) {
        acc_iterator t(*this);
        --pos_;
        return t;
    }
    acc_iterator& operator+=(difference_type n) {
        pos_ += n;
        return *this;
    }
    acc_iterator& operator-=(difference_type n) {
        pos_ -= n;
        return *this;
    }
    friend acc_iterator operator+(acc_iterator it, difference_type n) { return it += n; }
    friend acc_iterator operator+(difference_type n, acc_iterator it) { return it += n; }
    friend acc_iterator operator-(acc_iterator it, difference_type n) { return it -= n; }
    friend difference_type operator-(const acc_iterator& a, const acc_iterator& b) {
        return a.pos_ - b.pos_;
    }
    friend bool operator==(const acc_iterator& a, const acc_iterator& b) {
        return a.base_ == b.base_ && a.pos_ == b.pos_;
    }
    friend bool operator!=(const acc_iterator& a, const acc_iterator& b) { return !(a == b); }
    friend bool operator<(const acc_iterator& a, const acc_iterator& b) { return a.pos_ < b.pos_; }
    friend bool operator>(const acc_iterator& a, const acc_iterator& b) { return a.pos_ > b.pos_; }
    friend bool operator<=(const acc_iterator& a, const acc_iterator& b) { return a.pos_ <= b.pos_; }
    friend bool operator>=(const acc_iterator& a, const acc_iterator& b) { return a.pos_ >= b.pos_; }

private:
    template <typename, int>
    friend class acc_iterator;

    std::size_t address(difference_type p) const {
        std::size_t idx[Dims];
        std::size_t rem = static_cast<std::size_t>(p);
        for (int d = Dims - 1; d >= 0; --d) {
            const std::size_t extent = acc_[d] ? acc_[d] : 1;
            idx[d] = rem % extent;
            rem /= extent;
        }
        idx[0] += rem * (acc_[0] ? acc_[0] : 1);
        std::size_t lin = 0;
        for (int d = 0; d < Dims; ++d)
            lin = lin * buf_[d] + idx[d] + off_[d];
        return lin;
    }

    T* base_ = nullptr;
    range<Dims> buf_{};
    range<Dims> acc_{};
    id<Dims> off_{};
    difference_type pos_ = 0;
};

template <typename T, int Dims, int Level>
struct acc_ranged_subscript {
    T* base;
    range<Dims> buf;
    id<Dims> off;
    id<Dims> idx;
    decltype(auto) operator[](std::size_t i) const {
        id<Dims> next = idx;
        next[Level] = i;
        if constexpr (Level + 1 == Dims) {
            std::size_t lin = 0;
            for (int d = 0; d < Dims; ++d)
                lin = lin * buf[d] + next[d] + off[d];
            return (base[lin]);
        } else {
            return acc_ranged_subscript<T, Dims, Level + 1>{base, buf, off, next};
        }
    }
};

template <typename ElemT, int VD>
struct acc_storage {
    ElemT* base = nullptr;
    range<VD> buf_range{};
    range<VD> acc_range{};
    id<VD> offset{};
    property_list props{};

    std::size_t linear(const id<VD>& index) const {
        std::size_t lin = 0;
        for (int d = 0; d < VD; ++d)
            lin = lin * buf_range[d] + index[d] + offset[d];
        return lin;
    }
    std::size_t count() const {
        std::size_t n = 1;
        for (int d = 0; d < VD; ++d)
            n *= acc_range[d];
        return n;
    }
    bool operator==(const acc_storage& o) const {
        return base == o.base && buf_range == o.buf_range && acc_range == o.acc_range &&
               offset == o.offset;
    }
};

template <typename ElemT, int VD, typename BufT>
inline void acc_bind(acc_storage<ElemT, VD>& st, BufT& buf, const range<VD>& accessRange,
                     const id<VD>& accessOffset, const property_list& props, bool read_only,
                     bool check_range = true) {
    if (read_only && props.template paras_has_property<property::no_init>())
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "property::no_init is not allowed on a read-only accessor");
    const auto br = buf.get_range();
    range<VD> bufRange{};
    for (int d = 0; d < VD; ++d)
        bufRange[d] = br[d];
    for (int d = 0; d < VD && check_range; ++d)
        if (accessOffset[d] + accessRange[d] > bufRange[d])
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "accessor range + offset exceeds the buffer range");
    st.base = const_cast<ElemT*>(buf.data());
    st.buf_range = bufRange;
    st.acc_range = accessRange;
    st.offset = accessOffset;
    st.props = props;
}

template <int VD, typename BufT>
inline range<VD> acc_full_range(BufT& buf) {
    const auto br = buf.get_range();
    range<VD> r{};
    for (int d = 0; d < VD; ++d)
        r[d] = br[d];
    return r;
}
template <int VD>
inline range<VD> acc_one_range() {
    range<VD> r{};
    for (int d = 0; d < VD; ++d)
        r[d] = 1;
    return r;
}

} // namespace detail

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode = detail::default_acc_mode_v<DataT>,
          target AccessTarget = target::device,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class accessor {
    static_assert(Dimensions >= 0 && Dimensions <= 3, "accessor dimensions must be 0..3");
    static constexpr int VD = Dimensions == 0 ? 1 : Dimensions;
    static constexpr bool paras_read_only = AccessMode == access_mode::read;
    static constexpr bool paras_no_init_mode = AccessMode == access_mode::discard_write ||
                                               AccessMode == access_mode::discard_read_write;
    using elem_t = std::remove_const_t<DataT>;

    template <typename, int, access_mode, target, access::placeholder>
    friend class accessor;
    template <typename T>
    friend PARAS_KERNEL_HD auto detail::mp_acc_pointer(const T&) noexcept;
    template <int, typename>
    friend class image;
    template <typename, int, typename>
    friend class buffer;

public:
    using value_type = std::conditional_t<paras_read_only, const DataT, DataT>;
    using reference = value_type&;
    using const_reference = const DataT&;
    template <access::decorated IsDecorated>
    using accessor_ptr = multi_ptr<value_type, access::address_space::global_space, IsDecorated>;
    using iterator = detail::acc_iterator<value_type, VD>;
    using const_iterator = detail::acc_iterator<const value_type, VD>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = std::size_t;

    accessor() = default;

    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<D == 0>,
              int = 0>
    accessor(buffer<DataT, 1, AllocatorT>& bufferRef, const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<1>(bufferRef), id<1>{}, propList, true);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<D == 0>,
              int = 0>
    accessor(buffer<DataT, 1, AllocatorT>& bufferRef, handler&,
             const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<1>(bufferRef), id<1>{}, propList, false);
    }

    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef,
             const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<VD>(bufferRef), id<VD>{}, propList, true);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_acc_tag_v<DeductionTagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, DeductionTagT,
             const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<VD>(bufferRef), id<VD>{}, propList, true);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, handler&,
             const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<VD>(bufferRef), id<VD>{}, propList, false);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_acc_tag_v<DeductionTagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, handler&, DeductionTagT,
             const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<VD>(bufferRef), id<VD>{}, propList, false);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
             const property_list& propList = {}) {
        init(bufferRef, accessRange, id<VD>{}, propList, true);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_acc_tag_v<DeductionTagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
             DeductionTagT, const property_list& propList = {}) {
        init(bufferRef, accessRange, id<VD>{}, propList, true);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
             id<Dimensions> accessOffset, const property_list& propList = {}) {
        init(bufferRef, accessRange, accessOffset, propList, true);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_acc_tag_v<DeductionTagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
             id<Dimensions> accessOffset, DeductionTagT, const property_list& propList = {}) {
        init(bufferRef, accessRange, accessOffset, propList, true);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, handler&,
             range<Dimensions> accessRange, const property_list& propList = {}) {
        init(bufferRef, accessRange, id<VD>{}, propList, false);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_acc_tag_v<DeductionTagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, handler&,
             range<Dimensions> accessRange, DeductionTagT, const property_list& propList = {}) {
        init(bufferRef, accessRange, id<VD>{}, propList, false);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, handler&,
             range<Dimensions> accessRange, id<Dimensions> accessOffset,
             const property_list& propList = {}) {
        init(bufferRef, accessRange, accessOffset, propList, false);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_acc_tag_v<DeductionTagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, handler&,
             range<Dimensions> accessRange, id<Dimensions> accessOffset, DeductionTagT,
             const property_list& propList = {}) {
        init(bufferRef, accessRange, accessOffset, propList, false);
    }

    template <typename OtherT, access_mode OtherMode, access::placeholder OtherP,
              typename = std::enable_if_t<
                  AccessMode == access_mode::read &&
                  std::is_same_v<std::remove_const_t<OtherT>, elem_t> &&
                  ((OtherMode == access_mode::read_write && !std::is_const_v<OtherT>) ||
                   (OtherMode == access_mode::read && !std::is_same_v<OtherT, DataT>))>>
    accessor(const accessor<OtherT, Dimensions, OtherMode, AccessTarget, OtherP>& other)
        : st_(other.st_), placeholder_(other.placeholder_) {}

    void swap(accessor& other) {
        std::swap(st_, other.st_);
        std::swap(placeholder_, other.placeholder_);
    }

    bool is_placeholder() const {
        return AccessTarget == target::host_buffer ? false : placeholder_;
    }
    size_type byte_size() const noexcept { return size() * sizeof(DataT); }
    size_type size() const noexcept { return st_.count(); }
    size_type max_size() const noexcept { return size(); }
    std::size_t get_size() const noexcept { return byte_size(); }
    std::size_t get_count() const noexcept { return size(); }
    bool empty() const noexcept { return size() == 0; }

    template <int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    range<Dimensions> get_range() const {
        return st_.acc_range;
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    id<Dimensions> get_offset() const {
        return st_.offset;
    }

    template <int D = Dimensions,
              typename = std::enable_if_t<D == 0 && AccessMode != access_mode::atomic>>
    operator reference() const {
        return st_.base[st_.linear(id<VD>{})];
    }
    template <int D = Dimensions,
              typename = std::enable_if_t<D == 0 && AccessMode != access_mode::atomic &&
                                          AccessMode != access_mode::read>>
    const accessor& operator=(const value_type& other) const {
        st_.base[st_.linear(id<VD>{})] = other;
        return *this;
    }
    template <int D = Dimensions,
              typename = std::enable_if_t<D == 0 && AccessMode != access_mode::atomic &&
                                          AccessMode != access_mode::read>>
    const accessor& operator=(value_type&& other) const {
        st_.base[st_.linear(id<VD>{})] = std::move(other);
        return *this;
    }

    template <int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && AccessMode != access_mode::atomic>>
    reference operator[](id<Dimensions> index) const {
        return st_.base[st_.linear(index)];
    }
    template <typename IndexT, int D = Dimensions,
              typename = std::enable_if_t<D == 1 && std::is_integral_v<IndexT> &&
                                          AccessMode != access_mode::atomic>>
    reference operator[](IndexT index) const {
        return st_.base[st_.linear(id<1>(static_cast<std::size_t>(index)))];
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>, typename = void>
    auto operator[](std::size_t index) const {
        id<VD> idx{};
        idx[0] = index;
        return detail::acc_ranged_subscript<value_type, VD, 1>{st_.base, st_.buf_range, st_.offset,
                                                               idx};
    }

    template <int D = Dimensions,
              typename = std::enable_if_t<D == 0 && AccessMode == access_mode::atomic>>
    operator atomic<DataT, access::address_space::global_space>() const {
        return atomic<DataT, access::address_space::global_space>(
            multi_ptr<DataT, access::address_space::global_space, access::decorated::no>(
                st_.base + st_.linear(id<VD>{})));
    }
    template <int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && AccessMode == access_mode::atomic>,
              typename = void, typename = void>
    atomic<DataT, access::address_space::global_space> operator[](id<Dimensions> index) const {
        return atomic<DataT, access::address_space::global_space>(
            multi_ptr<DataT, access::address_space::global_space, access::decorated::no>(
                st_.base + st_.linear(index)));
    }
    template <int D = Dimensions,
              typename = std::enable_if_t<D == 1 && AccessMode == access_mode::atomic>,
              typename = void, typename = void, typename = void>
    atomic<DataT, access::address_space::global_space> operator[](std::size_t index) const {
        return (*this)[id<1>(index)];
    }

    auto get_pointer() const noexcept {
        if constexpr (AccessTarget == target::device) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            return global_ptr<value_type>(st_.base);
#pragma clang diagnostic pop
        } else if constexpr (AccessTarget == target::constant_buffer) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            return constant_ptr<DataT>(st_.base);
#pragma clang diagnostic pop
        } else {
            return static_cast<std::add_pointer_t<value_type>>(st_.base);
        }
    }
    template <access::decorated IsDecorated, target T = AccessTarget,
              typename = std::enable_if_t<T == target::device>>
    accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
        return accessor_ptr<IsDecorated>(static_cast<value_type*>(st_.base));
    }

    iterator begin() const noexcept {
        return iterator(st_.base, st_.buf_range, st_.acc_range, st_.offset, 0);
    }
    iterator end() const noexcept {
        return iterator(st_.base, st_.buf_range, st_.acc_range, st_.offset,
                        static_cast<difference_type>(size()));
    }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }
    reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    template <typename PropT>
    bool has_property() const noexcept {
        return st_.props.template paras_has_property<PropT>();
    }
    template <typename PropT>
    PropT get_property() const {
        return st_.props.template paras_get_property<PropT>();
    }

    friend bool operator==(const accessor& a, const accessor& b) { return a.st_ == b.st_; }
    friend bool operator!=(const accessor& a, const accessor& b) { return !(a == b); }

    template <typename CoordT>
    DataT read(const CoordT& coords, const sampler&) const {
        return (*this)[coords];
    }
    template <typename CoordT>
    void write(const CoordT& coords, const DataT& data) const {
        (*this)[coords] = data;
    }

    elem_t* paras_base() const noexcept { return st_.base; }

private:
    template <typename BufT>
    void init(BufT& buf, const range<VD>& r, const id<VD>& off, const property_list& props,
              bool placeholder) {
        if constexpr (AccessTarget == target::host_buffer)
            paras_drain_host_tasks();
        detail::acc_bind(st_, buf, r, off, props, paras_read_only, Dimensions != 0);
        placeholder_ = placeholder;
    }

    accessor(DataT* p, range<VD> rr, access_mode) {
        st_.base = const_cast<elem_t*>(p);
        st_.buf_range = rr;
        st_.acc_range = rr;
    }

    detail::acc_storage<elem_t, VD> st_{};
    bool placeholder_ = false;
};

#define PARAS_ACC_GUIDE(...)                                                                       \
    template <typename DataT, int Dimensions, typename AllocatorT, typename TagT,                  \
              typename = std::enable_if_t<detail::is_acc_tag_v<TagT>>>                             \
    accessor(buffer<DataT, Dimensions, AllocatorT>&, __VA_ARGS__ const property_list& = {})        \
        -> accessor<DataT, Dimensions, detail::acc_tag<TagT>::mode, detail::acc_tag<TagT>::tgt>;
PARAS_ACC_GUIDE(TagT, )
PARAS_ACC_GUIDE(handler&, TagT, )
PARAS_ACC_GUIDE(range<Dimensions>, TagT, )
PARAS_ACC_GUIDE(range<Dimensions>, id<Dimensions>, TagT, )
PARAS_ACC_GUIDE(handler&, range<Dimensions>, TagT, )
PARAS_ACC_GUIDE(handler&, range<Dimensions>, id<Dimensions>, TagT, )
#undef PARAS_ACC_GUIDE

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode = detail::default_acc_mode_v<DataT>>
class host_accessor {
    static_assert(Dimensions >= 0 && Dimensions <= 3, "host_accessor dimensions must be 0..3");
    static constexpr int VD = Dimensions == 0 ? 1 : Dimensions;
    static constexpr bool paras_read_only = AccessMode == access_mode::read;
    using elem_t = std::remove_const_t<DataT>;
    template <typename, int, access_mode>
    friend class host_accessor;

public:
    using value_type = std::conditional_t<paras_read_only, const DataT, DataT>;
    using reference = value_type&;
    using const_reference = const DataT&;
    using iterator = detail::acc_iterator<value_type, VD>;
    using const_iterator = detail::acc_iterator<const value_type, VD>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = std::size_t;

    host_accessor() = default;

    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<D == 0>,
              int = 0>
    host_accessor(buffer<DataT, 1, AllocatorT>& bufferRef, const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<1>(bufferRef), id<1>{}, propList);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    host_accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef,
                  const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<VD>(bufferRef), id<VD>{}, propList);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_host_acc_tag_v<DeductionTagT>>>
    host_accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, DeductionTagT,
                  const property_list& propList = {}) {
        init(bufferRef, detail::acc_full_range<VD>(bufferRef), id<VD>{}, propList);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    host_accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
                  const property_list& propList = {}) {
        init(bufferRef, accessRange, id<VD>{}, propList);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_host_acc_tag_v<DeductionTagT>>>
    host_accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
                  DeductionTagT, const property_list& propList = {}) {
        init(bufferRef, accessRange, id<VD>{}, propList);
    }
    template <typename AllocatorT, int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    host_accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
                  id<Dimensions> accessOffset, const property_list& propList = {}) {
        init(bufferRef, accessRange, accessOffset, propList);
    }
    template <typename AllocatorT, typename DeductionTagT, int D = Dimensions,
              typename = std::enable_if_t<(D > 0) && detail::is_host_acc_tag_v<DeductionTagT>>>
    host_accessor(buffer<DataT, Dimensions, AllocatorT>& bufferRef, range<Dimensions> accessRange,
                  id<Dimensions> accessOffset, DeductionTagT,
                  const property_list& propList = {}) {
        init(bufferRef, accessRange, accessOffset, propList);
    }

    template <typename OtherT, access_mode OtherMode,
              typename = std::enable_if_t<
                  AccessMode == access_mode::read &&
                  std::is_same_v<std::remove_const_t<OtherT>, elem_t> &&
                  ((OtherMode == access_mode::read_write && !std::is_const_v<OtherT>) ||
                   (OtherMode == access_mode::read && !std::is_same_v<OtherT, DataT>))>>
    host_accessor(const host_accessor<OtherT, Dimensions, OtherMode>& other) : st_(other.st_) {}

    void swap(host_accessor& other) { std::swap(st_, other.st_); }
    size_type byte_size() const noexcept { return size() * sizeof(DataT); }
    size_type size() const noexcept { return st_.count(); }
    size_type max_size() const noexcept { return size(); }
    bool empty() const noexcept { return size() == 0; }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    range<Dimensions> get_range() const {
        return st_.acc_range;
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    id<Dimensions> get_offset() const {
        return st_.offset;
    }
    template <int D = Dimensions, typename = std::enable_if_t<D == 0>>
    operator reference() const {
        return st_.base[st_.linear(id<VD>{})];
    }
    template <int D = Dimensions,
              typename = std::enable_if_t<D == 0 && AccessMode != access_mode::read>>
    const host_accessor& operator=(const value_type& other) const {
        st_.base[st_.linear(id<VD>{})] = other;
        return *this;
    }
    template <int D = Dimensions,
              typename = std::enable_if_t<D == 0 && AccessMode != access_mode::read>>
    const host_accessor& operator=(value_type&& other) const {
        st_.base[st_.linear(id<VD>{})] = std::move(other);
        return *this;
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    reference operator[](id<Dimensions> index) const {
        return st_.base[st_.linear(index)];
    }
    template <typename IndexT, int D = Dimensions,
              typename = std::enable_if_t<D == 1 && std::is_integral_v<IndexT>>>
    reference operator[](IndexT index) const {
        return st_.base[st_.linear(id<1>(static_cast<std::size_t>(index)))];
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>, typename = void>
    auto operator[](std::size_t index) const {
        id<VD> idx{};
        idx[0] = index;
        return detail::acc_ranged_subscript<value_type, VD, 1>{st_.base, st_.buf_range, st_.offset,
                                                               idx};
    }
    std::add_pointer_t<value_type> get_pointer() const noexcept { return st_.base; }

    iterator begin() const noexcept {
        return iterator(st_.base, st_.buf_range, st_.acc_range, st_.offset, 0);
    }
    iterator end() const noexcept {
        return iterator(st_.base, st_.buf_range, st_.acc_range, st_.offset,
                        static_cast<difference_type>(size()));
    }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }
    reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    template <typename PropT>
    bool has_property() const noexcept {
        return st_.props.template paras_has_property<PropT>();
    }
    template <typename PropT>
    PropT get_property() const {
        return st_.props.template paras_get_property<PropT>();
    }

    friend bool operator==(const host_accessor& a, const host_accessor& b) {
        return a.st_ == b.st_;
    }
    friend bool operator!=(const host_accessor& a, const host_accessor& b) { return !(a == b); }

    elem_t* paras_base() const noexcept { return st_.base; }

private:
    template <typename BufT>
    void init(BufT& buf, const range<VD>& r, const id<VD>& off, const property_list& props) {
        paras_drain_host_tasks();
        detail::acc_bind(st_, buf, r, off, props, paras_read_only, Dimensions != 0);
    }

    detail::acc_storage<elem_t, VD> st_{};
};

#define PARAS_HOST_ACC_GUIDE(...)                                                                  \
    template <typename DataT, int Dimensions, typename AllocatorT, typename TagT,                  \
              typename = std::enable_if_t<detail::is_host_acc_tag_v<TagT>>>                        \
    host_accessor(buffer<DataT, Dimensions, AllocatorT>&, __VA_ARGS__ const property_list& = {})   \
        -> host_accessor<DataT, Dimensions, detail::acc_tag<TagT>::mode>;
PARAS_HOST_ACC_GUIDE(TagT, )
PARAS_HOST_ACC_GUIDE(range<Dimensions>, TagT, )
PARAS_HOST_ACC_GUIDE(range<Dimensions>, id<Dimensions>, TagT, )
#undef PARAS_HOST_ACC_GUIDE

template <typename DataT, int Dimensions = 1>
class local_accessor {
public:
    using value_type = DataT;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = size_t;

    template <access::decorated IsDecorated>
    using accessor_ptr = multi_ptr<value_type, access::address_space::local_space, IsDecorated>;
    using iterator = value_type*;
    using const_iterator = const value_type*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;

    local_accessor() = default;

    template <typename OtherT,
              typename = std::enable_if_t<std::is_const_v<DataT> && !std::is_const_v<OtherT> &&
                                          std::is_same_v<const OtherT, DataT>>>
    PARAS_KERNEL_HD local_accessor(const local_accessor<OtherT, Dimensions>& other)
        : local_offset_bytes_(other.paras_offset()), ele_count_(other.paras_count()),
          props_(other.paras_props()) {}

    template <int D = Dimensions, typename std::enable_if_t<D == 0>* = nullptr>
    explicit local_accessor(handler& cghRef, const property_list& propList = {})
        : local_offset_bytes_(cghRef.template local_alloc_offset<DataT>(1)), ele_count_{},
          props_(propList) {}

    local_accessor(range<Dimensions> allocationRange, handler& cghRef,
                   const property_list& propList = {})
        : local_offset_bytes_(cghRef.template local_alloc_offset<DataT>(allocationRange.size())),
          ele_count_(allocationRange), props_(propList) {}

    void swap(local_accessor& other) {
        using std::swap;

        swap(local_offset_bytes_, other.local_offset_bytes_);
        swap(ele_count_, other.ele_count_);
        swap(props_, other.props_);
    }

    PARAS_KERNEL_HD
    size_type byte_size() const noexcept { return size() * sizeof(DataT); }

    PARAS_KERNEL_HD
    size_type size() const noexcept {
        if constexpr (Dimensions == 0) {
            return 1;
        } else {
            return ele_count_.size();
        }
    }

    PARAS_KERNEL_HD
    bool empty() const noexcept { return size() == 0; }

    template <int D = Dimensions, typename std::enable_if_t<(D > 0)>* = nullptr>
    PARAS_KERNEL_HD range<Dimensions> get_range() const {
        return ele_count_;
    }

    template <int D = Dimensions, typename T = DataT,
              typename = std::enable_if_t<D == 0 && !std::is_const_v<T>>>
    PARAS_KERNEL_HD const local_accessor& operator=(const value_type& val) const {
        *resolved_pointer() = val;
        return *this;
    }
    template <int D = Dimensions, typename T = DataT,
              typename = std::enable_if_t<D == 0 && !std::is_const_v<T>>>
    PARAS_KERNEL_HD const local_accessor& operator=(value_type&& val) const {
        *resolved_pointer() = std::move(val);
        return *this;
    }

    PARAS_KERNEL_HD size_type max_size() const noexcept { return size(); }
    PARAS_KERNEL_HD iterator begin() const noexcept { return resolved_pointer(); }
    PARAS_KERNEL_HD iterator end() const noexcept { return resolved_pointer() + size(); }
    PARAS_KERNEL_HD const_iterator cbegin() const noexcept { return begin(); }
    PARAS_KERNEL_HD const_iterator cend() const noexcept { return end(); }
    reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    friend bool operator==(const local_accessor& a, const local_accessor& b) {
        return a.local_offset_bytes_ == b.local_offset_bytes_ && a.ele_count_ == b.ele_count_;
    }
    friend bool operator!=(const local_accessor& a, const local_accessor& b) { return !(a == b); }

    size_type paras_offset() const noexcept { return local_offset_bytes_; }
    range<Dimensions> paras_count() const noexcept { return ele_count_; }
    property_list paras_props() const noexcept { return props_; }

    template <int D = Dimensions, typename std::enable_if_t<D == 0>* = nullptr>
    PARAS_KERNEL_HD operator reference() const {
        return *resolved_pointer();
    }

    template <int D = Dimensions, typename std::enable_if_t<(D > 0), int> = 0>
    PARAS_KERNEL_HD reference operator[](const id<Dimensions>& index) const {
        return resolved_pointer()[linear_id(index)];
    }

    template <typename IndexT, int D = Dimensions,
              typename std::enable_if_t<(D == 1) && std::is_integral_v<IndexT>, int> = 0>
    PARAS_KERNEL_HD reference operator[](IndexT index) const {
        return resolved_pointer()[static_cast<size_type>(index)];
    }

    template <int D = Dimensions, typename std::enable_if_t<(D > 1), int> = 0>
    PARAS_KERNEL_HD auto operator[](size_type index) const {
        return detail::accessor_subscript<DataT, Dimensions, 1>{resolved_pointer(), ele_count_,
                                                                index};
    }

    template <access::decorated IsDecorated>
    PARAS_KERNEL_HD accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
        return accessor_ptr<IsDecorated>{resolved_pointer()};
    }

    PARAS_KERNEL_HD
    local_ptr<value_type> get_pointer() const noexcept {
        return get_multi_ptr<access::decorated::legacy>();
    }

    PARAS_KERNEL_HD
    operator accessor_ptr<access::decorated::legacy>() const noexcept {
        return accessor_ptr<access::decorated::legacy>{resolved_pointer()};
    }

    PARAS_KERNEL_HD DataT* paras_base() const noexcept { return resolved_pointer(); }

private:
    size_type local_offset_bytes_ = 0;

    range<Dimensions> ele_count_{};
    property_list props_{};

    PARAS_KERNEL_HD
    DataT* resolved_pointer() const noexcept {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        unsigned char* const base = paras_get_dynamic_shared_memory();

        return reinterpret_cast<DataT*>(base + local_offset_bytes_);
#else
        if (paras_host_detail::tl_local_mem)
            return reinterpret_cast<DataT*>(paras_host_detail::tl_local_mem + local_offset_bytes_);
        return reinterpret_cast<DataT*>(paras_host_detail::host_local_scratch() +
                                        local_offset_bytes_);
#endif
    }

    PARAS_KERNEL_HD
    size_type linear_id(const id<Dimensions>& index) const noexcept {
        size_type linear = 0;

        for (int i = 0; i < Dimensions; ++i) {
            linear =
                linear * static_cast<size_type>(ele_count_[i]) + static_cast<size_type>(index[i]);
        }

        return linear;
    }
};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, target::local, IsPlaceholder>
    : public local_accessor<DataT, Dimensions> {
public:
    using local_accessor<DataT, Dimensions>::local_accessor;

    std::size_t get_size() const noexcept { return this->size() * sizeof(DataT); }
    std::size_t get_count() const noexcept { return this->size(); }
};

template <typename DataT, int Dimensions, access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, access_mode::atomic, target::local, IsPlaceholder>
    : public local_accessor<DataT, Dimensions> {
    using paras_base_t = local_accessor<DataT, Dimensions>;
    using paras_atomic_t = atomic<DataT, access::address_space::local_space>;
    using paras_mp_t = multi_ptr<DataT, access::address_space::local_space, access::decorated::no>;

public:
    using paras_base_t::paras_base_t;

    std::size_t get_size() const noexcept { return this->size() * sizeof(DataT); }
    std::size_t get_count() const noexcept { return this->size(); }

    template <int D = Dimensions, typename = std::enable_if_t<D == 0>>
    operator paras_atomic_t() const {
        DataT& r = static_cast<const paras_base_t&>(*this);
        return paras_atomic_t(paras_mp_t(&r));
    }
    template <int D = Dimensions, typename = std::enable_if_t<(D > 0)>>
    paras_atomic_t operator[](id<Dimensions> index) const {
        return paras_atomic_t(paras_mp_t(&paras_base_t::operator[](index)));
    }
    template <int D = Dimensions, typename = std::enable_if_t<D == 1>, typename = void>
    paras_atomic_t operator[](std::size_t index) const {
        return (*this)[id<1>(index)];
    }

    template <int Level>
    class paras_index {
    public:
        paras_index(const accessor& a, id<Dimensions> idx) : a_(a), idx_(idx) {}
        auto operator[](std::size_t i) const {
            id<Dimensions> n = idx_;
            n[Level] = i;
            if constexpr (Level + 1 == Dimensions)
                return a_[n];
            else
                return paras_index<Level + 1>(a_, n);
        }

    private:
        const accessor& a_;
        id<Dimensions> idx_;
    };
    template <int D = Dimensions, typename = std::enable_if_t<(D > 1)>, typename = void,
              typename = void>
    paras_index<1> operator[](std::size_t index) const {
        id<Dimensions> n{};
        n[0] = index;
        return paras_index<1>(*this, n);
    }
};

template <typename DataT, int Dimensions, access::mode AccessMode, target AccessTarget,
          access::placeholder IsPlaceholder>
struct is_property_of<property::no_init,
                      accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>>
    : std::true_type {};
template <typename DataT, int Dimensions, access::mode AccessMode>
struct is_property_of<property::no_init, host_accessor<DataT, Dimensions, AccessMode>>
    : std::true_type {};

namespace detail {
template <typename AccT>
PARAS_KERNEL_HD auto mp_acc_pointer(const AccT& acc) noexcept {
    return acc.paras_base();
}
} // namespace detail

} // namespace sycl

template <typename DataT, int Dimensions, sycl::access::mode AccessMode, sycl::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
struct std::hash<sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>> {
    std::size_t operator()(const sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                                                IsPlaceholder>& a) const noexcept {
        return std::hash<const void*>{}(a.paras_base());
    }
};
template <typename DataT, int Dimensions, sycl::access::mode AccessMode>
struct std::hash<sycl::host_accessor<DataT, Dimensions, AccessMode>> {
    std::size_t
    operator()(const sycl::host_accessor<DataT, Dimensions, AccessMode>& a) const noexcept {
        return std::hash<const void*>{}(a.paras_base());
    }
};

template <typename DataT, int Dimensions>
struct std::hash<sycl::local_accessor<DataT, Dimensions>> {
    std::size_t operator()(const sycl::local_accessor<DataT, Dimensions>& a) const noexcept {
        return std::hash<std::size_t>{}(a.paras_offset());
    }
};

#pragma clang diagnostic pop

#endif

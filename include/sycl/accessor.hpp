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

#include <iterator>
#include <vector>
#include "range.hpp"
#include "exception.hpp"
#include "id.hpp"
#include "item.hpp"
#include "access.hpp"
#include "property_list.hpp"
#include "handler.hpp"
#include "multi_ptr.hpp"
#include "local_ptr.hpp"
#include "sampler.hpp"
#include "../utilities/internal_utils.hpp"

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>
#include <limits>
#include <functional>

#include "kem_gpu/gpu_utilities.hpp"

namespace sycl
{
namespace detail {

template <int D>
inline void check_accessor_range(const range<D> &bufRange,
                                 const range<D> &accessRange,
                                 const id<D> &accessOffset = id<D>()) {
  for (int d = 0; d < D; ++d)
    if (accessRange[d] + accessOffset[d] > bufRange[d])
      throw exception(make_error_code(errc::invalid),
                      "accessor range + offset exceeds the buffer range");
}
}

    template <typename T, int Dimensions, typename AllocatorT>
    class buffer;

    template <typename T, access::address_space AddressSpace>
    class atomic;

    enum class target
    {
        device,
        host_task,
        host_buffer,

        global_buffer [[deprecated]] = device,
        constant_buffer [[deprecated]] = 100,
        local [[deprecated]] = 101,
        image [[deprecated]] = 102,
        host_image [[deprecated]] = 103,
        image_array [[deprecated]] = 104
    };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    namespace access
    {
        using target = ::sycl::target;
    }

    namespace detail
    {

        template <typename T, int D>
        class accessor_iterator
        {
            template <typename, int> friend class accessor_iterator;
            T *base_ = nullptr;
            range<D> mem_{};
            range<D> acc_{};
            id<D> off_{};
            std::ptrdiff_t lin_ = 0;

            PARAS_KERNEL_HD T *addr(std::ptrdiff_t lin) const
            {
                size_t rem = static_cast<size_t>(lin);
                size_t idx[D];
                for (int d = D - 1; d >= 1; --d)
                {
                    const size_t a = acc_[d] ? acc_[d] : 1;
                    idx[d] = rem % a + off_[d];
                    rem /= a;
                }
                idx[0] = rem + off_[0];
                size_t flat = 0;
                for (int d = 0; d < D; ++d)
                    flat = flat * mem_[d] + idx[d];
                return base_ + flat;
            }

        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type = std::remove_const_t<T>;
            using difference_type = std::ptrdiff_t;
            using pointer = T *;
            using reference = T &;

            accessor_iterator() = default;
            PARAS_KERNEL_HD accessor_iterator(T *base, range<D> mem, range<D> acc,
                                              id<D> off, std::ptrdiff_t lin)
                : base_(base), mem_(mem), acc_(acc), off_(off), lin_(lin) {}

            template <typename U, typename = std::enable_if_t<
                                      !std::is_same_v<U, T> &&
                                      std::is_same_v<const U, T>>>
            PARAS_KERNEL_HD accessor_iterator(const accessor_iterator<U, D> &o)
                : base_(o.base_), mem_(o.mem_), acc_(o.acc_), off_(o.off_), lin_(o.lin_) {}

            PARAS_KERNEL_HD reference operator*() const { return *addr(lin_); }
            PARAS_KERNEL_HD pointer operator->() const { return addr(lin_); }
            PARAS_KERNEL_HD reference operator[](difference_type n) const { return *addr(lin_ + n); }

            PARAS_KERNEL_HD accessor_iterator &operator++() { ++lin_; return *this; }
            PARAS_KERNEL_HD accessor_iterator operator++(int) { auto t = *this; ++lin_; return t; }
            PARAS_KERNEL_HD accessor_iterator &operator--() { --lin_; return *this; }
            PARAS_KERNEL_HD accessor_iterator operator--(int) { auto t = *this; --lin_; return t; }
            PARAS_KERNEL_HD accessor_iterator &operator+=(difference_type n) { lin_ += n; return *this; }
            PARAS_KERNEL_HD accessor_iterator &operator-=(difference_type n) { lin_ -= n; return *this; }
            PARAS_KERNEL_HD friend accessor_iterator operator+(accessor_iterator it, difference_type n) { it.lin_ += n; return it; }
            PARAS_KERNEL_HD friend accessor_iterator operator+(difference_type n, accessor_iterator it) { it.lin_ += n; return it; }
            PARAS_KERNEL_HD friend accessor_iterator operator-(accessor_iterator it, difference_type n) { it.lin_ -= n; return it; }
            PARAS_KERNEL_HD friend difference_type operator-(const accessor_iterator &a, const accessor_iterator &b) { return a.lin_ - b.lin_; }
            PARAS_KERNEL_HD friend bool operator==(const accessor_iterator &a, const accessor_iterator &b) { return a.base_ == b.base_ && a.lin_ == b.lin_; }
            PARAS_KERNEL_HD friend bool operator!=(const accessor_iterator &a, const accessor_iterator &b) { return !(a == b); }
            PARAS_KERNEL_HD friend bool operator<(const accessor_iterator &a, const accessor_iterator &b) { return a.lin_ < b.lin_; }
            PARAS_KERNEL_HD friend bool operator>(const accessor_iterator &a, const accessor_iterator &b) { return a.lin_ > b.lin_; }
            PARAS_KERNEL_HD friend bool operator<=(const accessor_iterator &a, const accessor_iterator &b) { return a.lin_ <= b.lin_; }
            PARAS_KERNEL_HD friend bool operator>=(const accessor_iterator &a, const accessor_iterator &b) { return a.lin_ >= b.lin_; }
        };
    }

    namespace detail {

    template <typename T, typename H>
    std::size_t paras_legacy_local_off(H &cgh, std::size_t n) {
        const std::size_t off = cgh.template legacy_local_alloc_offset<T>(n);
#if !PARAS_GPU_BACKEND
        if (off + (n ? n : 1) * sizeof(T) > paras_host_detail::kHostLocalMemBytes)
            return static_cast<std::size_t>(-1);
#endif
        return off;
    }
    template <typename T> struct paras_acc_ptr {
        T *p = nullptr;
        std::size_t local_off = static_cast<std::size_t>(-1);
        paras_acc_ptr() = default;
        PARAS_KERNEL_HD paras_acc_ptr(T *q) noexcept : p(q) {}
        template <typename U>
        PARAS_KERNEL_HD static paras_acc_ptr from(const paras_acc_ptr<U> &o) noexcept {
            paras_acc_ptr r(const_cast<T *>(static_cast<const T *>(o.p)));
            r.local_off = o.local_off;
            return r;
        }
        PARAS_KERNEL_HD paras_acc_ptr &operator=(T *q) noexcept {
            p = q;
            local_off = static_cast<std::size_t>(-1);
            return *this;
        }
        PARAS_KERNEL_HD T *get() const noexcept {
    #if PARAS_DEVICE_PASS
            if (local_off != static_cast<std::size_t>(-1))
                return reinterpret_cast<T *>(paras_get_dynamic_shared_memory() + local_off);
    #else

            if (local_off != static_cast<std::size_t>(-1) && paras_host_detail::tl_local_mem)
                return reinterpret_cast<T *>(paras_host_detail::tl_local_mem + local_off);
    #endif
            return p;
        }
        PARAS_KERNEL_HD operator T *() const noexcept { return get(); }
    };
    }

#if !PARAS_NO_BUFFERS
    template <
    typename DataT,
    int Dimensions = 1,
    access::mode AccessMode = access::mode::read_write,
    target AccessTarget = target::device,
    access::placeholder IsPlaceholder = access::placeholder::false_t>
    class accessor
    {
        mutable detail::paras_acc_ptr<DataT> ptr_;
        range<Dimensions> r_;
        range<Dimensions> mem_r_{};
        id<Dimensions> offset_{};
        bool no_init_ = false;
        bool placeholder_ = false;

        void paras_check_no_init() const
        {
            if (no_init_ && AccessMode == access::mode::read)
                throw sycl::exception(make_error_code(errc::invalid),
                                      "accessor: no_init with access_mode::read");
        }

    public:

        template <typename, int, access::mode, target, access::placeholder>
        friend class accessor;
        template <typename SrcT, access::mode SrcMode,
                  typename = std::enable_if_t<
                      std::is_same_v<std::remove_const_t<SrcT>, std::remove_const_t<DataT>> &&
                      !(std::is_same_v<SrcT, DataT> && SrcMode == AccessMode) &&
                      AccessMode == access::mode::read &&
                      (SrcMode == access::mode::read ||
                       (SrcMode == access::mode::read_write && std::is_const_v<DataT>))>>
        accessor(const accessor<SrcT, Dimensions, SrcMode, AccessTarget, IsPlaceholder> &o)
            : ptr_(detail::paras_acc_ptr<DataT>::from(o.ptr_)), r_(o.r_), mem_r_(o.mem_r_), offset_(o.offset_), no_init_(o.no_init_), placeholder_(o.placeholder_) {}
    private:

    public:

        using value_type = std::conditional_t<AccessMode == access::mode::read, const DataT, DataT>;
        using reference = value_type &;
        using const_reference = const DataT &;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        template <access::decorated IsDecorated2>
        using accessor_ptr = multi_ptr<value_type, access::address_space::global_space, IsDecorated2>;

        using iterator = detail::accessor_iterator<value_type, Dimensions>;
        using const_iterator = detail::accessor_iterator<const DataT, Dimensions>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        accessor(DataT *p, range<Dimensions> rr, access::mode)
            : ptr_(p), r_(rr), mem_r_(rr) {}

        accessor() : ptr_(nullptr), r_() {}

        accessor(range<Dimensions> allocationSize, handler &paras_cgh,
                 const property_list &propList = {})
            : ptr_(paras_raw_zeroed(allocationSize.size())), r_(allocationSize), mem_r_(allocationSize)
        {
            no_init_ = propList.has_no_init();
            if constexpr (AccessTarget == target::local)
                ptr_.local_off = detail::paras_legacy_local_off<std::remove_const_t<DataT>>(paras_cgh, allocationSize.size());
        }

        static DataT *paras_raw_zeroed(size_t n)
        {
            void *p = ::operator new[]((n ? n : 1) * sizeof(DataT));
            std::memset(p, 0, (n ? n : 1) * sizeof(DataT));
            return static_cast<DataT *>(p);
        }

        template <access::decorated DecorateAddress>
        auto get_multi_ptr() const
        {
            if constexpr (AccessTarget == target::local)
            {
                return address_space_cast<access::address_space::local_space, DecorateAddress>(static_cast<value_type *>(ptr_));
            }
            else
            {
                return address_space_cast<access::address_space::global_space, DecorateAddress>(static_cast<value_type *>(ptr_));
            }
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <target T2 = AccessTarget, typename = std::enable_if_t<T2 == target::local>>
        accessor(range<Dimensions> allocationSize, handler &paras_cgh,
                 const property_list &props = {})
            : ptr_(new DataT[allocationSize.size()]()), r_(allocationSize), mem_r_(allocationSize)
        {
            no_init_ = props.has_no_init();
            if constexpr (AccessTarget == target::local)
                ptr_.local_off = detail::paras_legacy_local_off<std::remove_const_t<DataT>>(paras_cgh, allocationSize.size());
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 TagT, const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, TagT, const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange, id<Dimensions> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange, id<Dimensions> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange, TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        PARAS_KERNEL_HD DataT *paras_raw() const noexcept { return ptr_; }

        auto get_pointer() const
        {
            using elem_t = std::remove_const_t<DataT>;
            if constexpr (AccessTarget == target::constant_buffer)
                return multi_ptr<elem_t, access::address_space::constant_space,
                                 access::decorated::legacy>(const_cast<elem_t *>(ptr_.get()));
            else if constexpr (AccessTarget == target::local)
                return multi_ptr<elem_t, access::address_space::local_space,
                                 access::decorated::legacy>(const_cast<elem_t *>(ptr_.get()));
            else if constexpr (AccessTarget == target::device)
                return multi_ptr<value_type, access::address_space::global_space,
                                 access::decorated::legacy>(static_cast<value_type *>(ptr_));
            else
                return static_cast<value_type *>(ptr_);
        }

        iterator begin() const { return iterator(ptr_, mem_r_, r_, offset_, 0); }
        iterator end() const
        {
            return iterator(ptr_, mem_r_, r_, offset_, static_cast<std::ptrdiff_t>(r_.size()));
        }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }
        reverse_iterator rbegin() const { return reverse_iterator(end()); }
        reverse_iterator rend() const { return reverse_iterator(begin()); }
        const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
        const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

        template <typename PropT>
        bool has_property() const noexcept
        {
            if constexpr (std::is_same_v<PropT, property::no_init>)
                return no_init_;
            else
                return false;
        }

        bool is_placeholder() const noexcept { return placeholder_; }

        size_t size() const noexcept { return r_.size(); }

        bool empty() const noexcept { return size() == 0; }

        size_t byte_size() const noexcept { return size() * sizeof(DataT); }

        size_t max_size() const noexcept
        {
            return std::numeric_limits<size_t>::max() / sizeof(DataT);
        }

        template <typename PropT>
        PropT get_property() const
        {
            static_assert(std::is_same_v<PropT, property::no_init>,
                          "accessor::get_property: unsupported property type");
            if (!has_property<PropT>())
                throw sycl::exception(make_error_code(errc::invalid),
                                      "accessor::get_property: property not present");
            return property::no_init{};
        }

        void swap(accessor &other) noexcept
        {
            using std::swap;
            swap(ptr_, other.ptr_);
            swap(r_, other.r_);
            swap(mem_r_, other.mem_r_);
            swap(offset_, other.offset_);
            swap(no_init_, other.no_init_);
        }

        friend bool operator==(const accessor &lhs, const accessor &rhs) noexcept
        {
            return lhs.ptr_ == rhs.ptr_ &&
                   lhs.r_.size() == rhs.r_.size() &&
                   [&] {
                       for (int i = 0; i < Dimensions; ++i)
                           if (lhs.offset_[i] != rhs.offset_[i])
                               return false;
                       return true;
                   }();
        }

        friend bool operator!=(const accessor &lhs, const accessor &rhs) noexcept
        {
            return !(lhs == rhs);
        }

        size_t get_count() const noexcept { return size(); }

        size_t get_size() const noexcept { return byte_size(); }

        range<Dimensions> get_range() const noexcept { return r_; }

        id<Dimensions> get_offset() const noexcept { return offset_; }

        class row
        {
            mutable DataT *base_;

        public:
            PARAS_KERNEL_HD
            row(DataT *p) : base_(p) {}
            reference operator[](size_t i) const { return base_[i]; }
        };

        class plane
        {
            mutable DataT *base_;
            size_t stride1_;

        public:
            PARAS_KERNEL_HD
            plane(DataT *p, size_t stride1) : base_(p), stride1_(stride1) {}
            PARAS_KERNEL_HD
            row operator[](size_t i) const { return row(base_ + i * stride1_); }
        };

        decltype(auto) operator[](size_t i) const
        {
            if constexpr (Dimensions == 1) {
                return static_cast<reference>(ptr_[offset_[0] + i]);
            } else if constexpr (Dimensions == 2) {
                return row(ptr_ + (offset_[0] + i) * mem_r_[1] + offset_[1]);
            } else {
                return plane(ptr_ + (offset_[0] + i) * mem_r_[1] * mem_r_[2] + offset_[1] * mem_r_[2] + offset_[2], mem_r_[2]);
            }
        }

        reference operator[](id<Dimensions> idx) const
        {
            if constexpr (Dimensions == 1)
                return ptr_[offset_[0] + idx[0]];
            else if constexpr (Dimensions == 2)
                return ptr_[(offset_[0] + idx[0]) * mem_r_[1] + (offset_[1] + idx[1])];
            else
                return ptr_[((offset_[0] + idx[0]) * mem_r_[1] + (offset_[1] + idx[1])) * mem_r_[2] + (offset_[2] + idx[2])];
        }

        template <bool WithOffset>
        reference operator[](item<Dimensions, WithOffset> idx) const {
            return (*this)[idx.get_id()];
        }

        template <typename CoordT>
        DataT read(const CoordT &coords, const sampler &) const
        {
            return (*this)[coords];
        }

        template <typename CoordT>
        void write(const CoordT &coords, const DataT &data) const
        {
            (*this)[coords] = data;
        }
    };

    template <typename DataT, int Dimensions, typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, read_only_tag_t, const property_list & = {})
        -> accessor<DataT, Dimensions, access::mode::read, target::device>;

    template <typename DataT, int Dimensions, typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, write_only_tag_t, const property_list & = {})
        -> accessor<DataT, Dimensions, access::mode::write, target::device>;

    template <typename DataT, int Dimensions, typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, read_write_tag_t, const property_list & = {})
        -> accessor<DataT, Dimensions, access::mode::read_write, target::device>;

    template <typename DataT, int Dimensions, typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, read_only_host_task_tag_t, const property_list & = {})
        -> accessor<DataT, Dimensions, access::mode::read, target::host_task>;

    template <typename DataT, int Dimensions, typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, write_only_host_task_tag_t, const property_list & = {})
        -> accessor<DataT, Dimensions, access::mode::write, target::host_task>;

    template <typename DataT, int Dimensions, typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, read_write_host_task_tag_t, const property_list & = {})
        -> accessor<DataT, Dimensions, access::mode::read_write, target::host_task>;

    template <typename DataT,
              access::mode AccessMode,
              target AccessTarget>
    class accessor<DataT, 2, AccessMode, AccessTarget>
    {
        mutable detail::paras_acc_ptr<DataT> ptr_;
        range<2> r_;
        range<2> mem_r_{};
        id<2> offset_{};
        bool no_init_ = false;
        bool placeholder_ = false;

        void paras_check_no_init() const
        {
            if (no_init_ && AccessMode == access::mode::read)
                throw sycl::exception(make_error_code(errc::invalid),
                                      "accessor: no_init with access_mode::read");
        }

    public:

        template <typename, int, access::mode, target, access::placeholder>
        friend class accessor;
        template <typename SrcT, access::mode SrcMode,
                  typename = std::enable_if_t<
                      std::is_same_v<std::remove_const_t<SrcT>, std::remove_const_t<DataT>> &&
                      !(std::is_same_v<SrcT, DataT> && SrcMode == AccessMode) &&
                      AccessMode == access::mode::read &&
                      (SrcMode == access::mode::read ||
                       (SrcMode == access::mode::read_write && std::is_const_v<DataT>))>>
        accessor(const accessor<SrcT, 2, SrcMode, AccessTarget, access::placeholder::false_t> &o)
            : ptr_(detail::paras_acc_ptr<DataT>::from(o.ptr_)), r_(o.r_), mem_r_(o.mem_r_), offset_(o.offset_), no_init_(o.no_init_), placeholder_(o.placeholder_) {}
    private:

    public:

        using value_type = std::conditional_t<AccessMode == access::mode::read, const DataT, DataT>;
        using reference = value_type &;
        using const_reference = const DataT &;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        template <access::decorated IsDecorated2>
        using accessor_ptr = multi_ptr<value_type, access::address_space::global_space, IsDecorated2>;

        using iterator = detail::accessor_iterator<value_type, 2>;
        using const_iterator = detail::accessor_iterator<const DataT, 2>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        accessor() : ptr_(nullptr), r_(range<2>(0, 0)) {}

        accessor(DataT *p, range<2> rr, access::mode)
            : ptr_(p), r_(rr), mem_r_(rr) {}

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <target T2 = AccessTarget, typename = std::enable_if_t<T2 == target::local>>
        accessor(range<2> allocationSize, handler &paras_cgh,
                 const property_list &props = {})
            : ptr_(new DataT[allocationSize.size()]()), r_(allocationSize), mem_r_(allocationSize)
        {
            no_init_ = props.has_no_init();
            if constexpr (AccessTarget == target::local)
                ptr_.local_off = detail::paras_legacy_local_off<std::remove_const_t<DataT>>(paras_cgh, allocationSize.size());
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range()), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange, id<2> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange, id<2> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange, id<2> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange, TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange, id<2> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range()), offset_(accessOffset)
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), mem_r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        PARAS_KERNEL_HD DataT *paras_raw() const noexcept { return ptr_; }

        auto get_pointer() const
        {
            using elem_t = std::remove_const_t<DataT>;
            if constexpr (AccessTarget == target::constant_buffer)
                return multi_ptr<elem_t, access::address_space::constant_space,
                                 access::decorated::legacy>(const_cast<elem_t *>(ptr_.get()));
            else if constexpr (AccessTarget == target::local)
                return multi_ptr<elem_t, access::address_space::local_space,
                                 access::decorated::legacy>(const_cast<elem_t *>(ptr_.get()));
            else if constexpr (AccessTarget == target::device)
                return multi_ptr<value_type, access::address_space::global_space,
                                 access::decorated::legacy>(static_cast<value_type *>(ptr_));
            else
                return static_cast<value_type *>(ptr_);
        }

        template <access::decorated DecorateAddress>
        auto get_multi_ptr() const
        {
            return address_space_cast<access::address_space::global_space, DecorateAddress>(static_cast<value_type *>(ptr_));
        }

        iterator begin() const { return iterator(ptr_, mem_r_, r_, offset_, 0); }
        iterator end() const
        {
            return iterator(ptr_, mem_r_, r_, offset_, static_cast<std::ptrdiff_t>(r_.size()));
        }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }
        reverse_iterator rbegin() const { return reverse_iterator(end()); }
        reverse_iterator rend() const { return reverse_iterator(begin()); }
        const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
        const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

        template <typename PropT>
        bool has_property() const noexcept
        {
            if constexpr (std::is_same_v<PropT, property::no_init>)
                return no_init_;
            else
                return false;
        }

        bool is_placeholder() const noexcept { return placeholder_; }

        size_t size() const noexcept { return r_.size(); }

        bool empty() const noexcept { return size() == 0; }

        size_t byte_size() const noexcept { return size() * sizeof(DataT); }

        size_t max_size() const noexcept
        {
            return std::numeric_limits<size_t>::max() / sizeof(DataT);
        }

        template <typename PropT>
        PropT get_property() const
        {
            static_assert(std::is_same_v<PropT, property::no_init>,
                          "accessor::get_property: unsupported property type");
            if (!has_property<PropT>())
                throw sycl::exception(make_error_code(errc::invalid),
                                      "accessor::get_property: property not present");
            return property::no_init{};
        }

        void swap(accessor &other) noexcept
        {
            using std::swap;
            swap(ptr_, other.ptr_);
            swap(r_, other.r_);
            swap(mem_r_, other.mem_r_);
            swap(offset_, other.offset_);
            swap(no_init_, other.no_init_);
        }

        friend bool operator==(const accessor &lhs, const accessor &rhs) noexcept
        {
            return lhs.ptr_ == rhs.ptr_ &&
                   lhs.r_.size() == rhs.r_.size() &&
                   lhs.offset_[0] == rhs.offset_[0] &&
                   lhs.offset_[1] == rhs.offset_[1];
        }

        friend bool operator!=(const accessor &lhs, const accessor &rhs) noexcept
        {
            return !(lhs == rhs);
        }

        size_t get_count() const noexcept { return size(); }

        size_t get_size() const noexcept { return byte_size(); }

        range<2> get_range() const noexcept { return r_; }

        id<2> get_offset() const noexcept { return offset_; }

        class row
        {
            mutable DataT *base_;

        public:
            PARAS_KERNEL_HD
            row(DataT *p) : base_(p) {}
            reference operator[](size_t i) const { return base_[i]; }
        };

        row operator[](size_t j) const
        {
            return row(ptr_ + (offset_[0] + j) * mem_r_[1] + offset_[1]);
        }

        reference operator[](id<2> idx) const
        {
            return ptr_[(offset_[0] + idx[0]) * mem_r_[1] + (offset_[1] + idx[1])];
        }

        template <typename CoordT>
        DataT read(const CoordT &coords, const sampler &) const
        {
            return (*this)[coords];
        }

        template <typename CoordT>
        void write(const CoordT &coords, const DataT &data) const
        {
            (*this)[coords] = data;
        }
    };

    template <typename DataT,
              access::mode AccessMode,
              target AccessTarget,
              access::placeholder IsPlaceholder>
    class accessor<DataT, 0, AccessMode, AccessTarget, IsPlaceholder>
    {
        mutable detail::paras_acc_ptr<DataT> ptr_;
        bool no_init_ = false;
        bool placeholder_ = false;

        void paras_check_no_init() const
        {
            if (no_init_ && AccessMode == access::mode::read)
                throw sycl::exception(make_error_code(errc::invalid),
                                      "accessor: no_init with access_mode::read");
        }

    public:

        template <typename, int, access::mode, target, access::placeholder>
        friend class accessor;
        template <typename SrcT, access::mode SrcMode,
                  typename = std::enable_if_t<
                      std::is_same_v<std::remove_const_t<SrcT>, std::remove_const_t<DataT>> &&
                      !(std::is_same_v<SrcT, DataT> && SrcMode == AccessMode) &&
                      AccessMode == access::mode::read &&
                      (SrcMode == access::mode::read ||
                       (SrcMode == access::mode::read_write && std::is_const_v<DataT>))>>
        accessor(const accessor<SrcT, 0, SrcMode, AccessTarget, IsPlaceholder> &o)
            : ptr_(detail::paras_acc_ptr<DataT>::from(o.ptr_)), no_init_(o.no_init_), placeholder_(o.placeholder_) {}
    private:

    public:

        using value_type = std::conditional_t<AccessMode == access::mode::read, const DataT, DataT>;
        using reference = value_type &;
        using const_reference = const DataT &;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        template <access::decorated IsDecorated2>
        using accessor_ptr = multi_ptr<value_type, access::address_space::global_space, IsDecorated2>;

        using iterator = DataT *;
        using const_iterator = const value_type *;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        accessor() : ptr_(nullptr) {}

        template <typename AllocatorT>
        accessor(buffer<DataT, 1, AllocatorT> &paras_buf,
                 const property_list &props = {})
            : ptr_(paras_buf.size() ? paras_buf.data() : nullptr)
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 1, AllocatorT> &paras_buf,
                 handler &, const property_list &props = {})
            : ptr_(paras_buf.size() ? paras_buf.data() : nullptr)
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <target T2 = AccessTarget, typename = std::enable_if_t<T2 == target::local>>
        accessor(handler &paras_cgh, const property_list &props = {})
            : ptr_(new DataT[1]())
        {
            no_init_ = props.has_no_init();
            if constexpr (AccessTarget == target::local)
                ptr_.local_off = detail::paras_legacy_local_off<std::remove_const_t<DataT>>(paras_cgh, 1);
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 1, AllocatorT> &paras_buf,
                 TagT, const property_list &props = {})
            : ptr_(paras_buf.size() ? paras_buf.data() : nullptr)
        {
            placeholder_ = AccessTarget != target::host_buffer;
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (AccessTarget == target::host_buffer)
                ::sycl::detail::buffer_host_access_sync();
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 1, AllocatorT> &paras_buf,
                 handler &, TagT, const property_list &props = {})
            : ptr_(paras_buf.size() ? paras_buf.data() : nullptr)
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
            {
                no_init_ = true;
                paras_check_no_init();
            }
            else
                no_init_ = props.has_no_init();
            paras_check_no_init();
        }

        PARAS_KERNEL_HD DataT *paras_raw() const noexcept { return ptr_; }

        auto get_pointer() const
        {
            using elem_t = std::remove_const_t<DataT>;
            if constexpr (AccessTarget == target::constant_buffer)
                return multi_ptr<elem_t, access::address_space::constant_space,
                                 access::decorated::legacy>(const_cast<elem_t *>(ptr_.get()));
            else if constexpr (AccessTarget == target::local)
                return multi_ptr<elem_t, access::address_space::local_space,
                                 access::decorated::legacy>(const_cast<elem_t *>(ptr_.get()));
            else if constexpr (AccessTarget == target::device)
                return multi_ptr<value_type, access::address_space::global_space,
                                 access::decorated::legacy>(static_cast<value_type *>(ptr_));
            else
                return static_cast<value_type *>(ptr_);
        }

        template <access::decorated DecorateAddress>
        auto get_multi_ptr() const
        {
            return address_space_cast<access::address_space::global_space, DecorateAddress>(static_cast<value_type *>(ptr_));
        }

        operator reference() const { return *ptr_; }
        reference operator*() const { return *ptr_; }
        value_type *operator->() const { return ptr_; }

        operator atomic<DataT, access::address_space::local_space>() const {
            return atomic<DataT, access::address_space::local_space>(*ptr_);
        }

        template <access::mode M = AccessMode,
                  typename = std::enable_if_t<M != access::mode::read>>
        const accessor &operator=(const DataT &value) const
        {
            *ptr_ = value;
            return *this;
        }
        template <access::mode M = AccessMode,
                  typename = std::enable_if_t<M != access::mode::read>>
        const accessor &operator=(DataT &&value) const
        {
            *ptr_ = std::move(value);
            return *this;
        }

        template <typename PropT>
        bool has_property() const noexcept
        {
            if constexpr (std::is_same_v<PropT, property::no_init>)
                return no_init_;
            else
                return false;
        }

        bool is_placeholder() const noexcept { return placeholder_; }

        size_t size() const noexcept { return ptr_ ? 1 : 0; }

        bool empty() const noexcept { return ptr_ == nullptr; }

        size_t byte_size() const noexcept { return size() * sizeof(DataT); }

        size_t max_size() const noexcept { return 1; }

        DataT *begin() const { return ptr_; }
        DataT *end() const { return ptr_ + size(); }

        const DataT *cbegin() const { return ptr_; }
        const DataT *cend() const { return ptr_ + size(); }

        std::reverse_iterator<DataT *> rbegin() const
        {
            return std::reverse_iterator<DataT *>(end());
        }

        std::reverse_iterator<DataT *> rend() const
        {
            return std::reverse_iterator<DataT *>(begin());
        }

        std::reverse_iterator<const DataT *> crbegin() const
        {
            return std::reverse_iterator<const DataT *>(cend());
        }

        std::reverse_iterator<const DataT *> crend() const
        {
            return std::reverse_iterator<const DataT *>(cbegin());
        }

        template <typename PropT>
        PropT get_property() const
        {
            static_assert(std::is_same_v<PropT, property::no_init>,
                          "accessor::get_property: unsupported property type");
            if (!has_property<PropT>())
                throw sycl::exception(make_error_code(errc::invalid),
                                      "accessor::get_property: property not present");
            return property::no_init{};
        }

        void swap(accessor &other) noexcept
        {
            using std::swap;
            swap(ptr_, other.ptr_);
            swap(no_init_, other.no_init_);
        }

        friend bool operator==(const accessor &lhs, const accessor &rhs) noexcept
        {
            return lhs.ptr_ == rhs.ptr_;
        }

        friend bool operator!=(const accessor &lhs, const accessor &rhs) noexcept
        {
            return !(lhs == rhs);
        }

        size_t get_count() const noexcept { return size(); }

        size_t get_size() const noexcept { return byte_size(); }
    };

    template <typename DataT, int Dimensions = 1, access::mode AccessMode = access::mode::read_write>
    class host_accessor
        : public accessor<DataT, Dimensions, AccessMode, target::host_buffer> {
        using base_t = accessor<DataT, Dimensions, AccessMode, target::host_buffer>;

    public:
        using base_t::base_t;
        host_accessor() = default;
        host_accessor(const base_t &other) : base_t(other) {}

        template <access::mode M = AccessMode, int D = Dimensions,
                  typename = std::enable_if_t<M != access::mode::read && D == 0>>
        const host_accessor &operator=(const DataT &value) const
        {
            base_t::operator=(value);
            return *this;
        }
        template <access::mode M = AccessMode, int D = Dimensions,
                  typename = std::enable_if_t<M != access::mode::read && D == 0>>
        const host_accessor &operator=(DataT &&value) const
        {
            base_t::operator=(std::move(value));
            return *this;
        }
    };

    template <typename T, int D, typename A>
    host_accessor(buffer<T, D, A> &, const property_list & = {})
        -> host_accessor<T, D, access::mode::read_write>;
    template <typename T, int D, typename A, typename TagT,
              typename = decltype(TagT::value)>
    host_accessor(buffer<T, D, A> &, TagT, const property_list & = {})
        -> host_accessor<T, D, TagT::value>;
    template <typename T, int D, typename A>
    host_accessor(buffer<T, D, A> &, range<D>, const property_list & = {})
        -> host_accessor<T, D, access::mode::read_write>;
    template <typename T, int D, typename A, typename TagT,
              typename = decltype(TagT::value)>
    host_accessor(buffer<T, D, A> &, range<D>, TagT, const property_list & = {})
        -> host_accessor<T, D, TagT::value>;
    template <typename T, int D, typename A>
    host_accessor(buffer<T, D, A> &, range<D>, id<D>, const property_list & = {})
        -> host_accessor<T, D, access::mode::read_write>;
    template <typename T, int D, typename A, typename TagT,
              typename = decltype(TagT::value)>
    host_accessor(buffer<T, D, A> &, range<D>, id<D>, TagT,
                  const property_list & = {})
        -> host_accessor<T, D, TagT::value>;
#endif

    template <typename DataT, int Dimensions = 1>
    class local_accessor
    {
    public:

        template <typename, int> friend class local_accessor;
        template <typename SrcT,
                  typename = std::enable_if_t<std::is_const_v<DataT> &&
                      std::is_same_v<SrcT, std::remove_const_t<DataT>>>>
        local_accessor(const local_accessor<SrcT, Dimensions> &o)
            : local_offset_bytes_(o.local_offset_bytes_),
              ele_count_(o.ele_count_),
              host_storage_(o.host_storage_), props_(o.props_) {}

        using value_type = DataT;
        using reference = value_type &;
        using const_reference = const value_type &;
        using size_type = size_t;

        using iterator = DataT *;
        using const_iterator = const DataT *;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        template <access::decorated IsDecorated>
        using accessor_ptr =
            multi_ptr<
                value_type,
                access::address_space::local_space,
                IsDecorated>;

        local_accessor() = default;

        template <
            int D = Dimensions,
            typename std::enable_if_t<D == 0> * = nullptr>
        explicit local_accessor(
            handler &cghRef,
            const property_list &propList = {})
            : local_offset_bytes_(
                  cghRef.template local_alloc_offset<DataT>(1)),
              ele_count_{},

              host_storage_(static_cast<DataT *>(::operator new[](1 * sizeof(DataT)))),
              props_(propList)
        {
        }

        local_accessor(
            range<Dimensions> allocationRange,
            handler &cghRef,
            const property_list &propList = {})
            : local_offset_bytes_(
                  cghRef.template local_alloc_offset<DataT>(
                      allocationRange.size())),
              ele_count_(allocationRange),

              host_storage_(static_cast<DataT *>(::operator new[](allocationRange.size() * sizeof(DataT)))),
              props_(propList)
        {
        }

        void swap(local_accessor &other)
        {
            using std::swap;

            swap(local_offset_bytes_, other.local_offset_bytes_);
            swap(ele_count_, other.ele_count_);
            swap(host_storage_, other.host_storage_);
            swap(props_, other.props_);
        }

        size_type byte_size() const noexcept
        {
            return size() * sizeof(DataT);
        }

        size_type size() const noexcept
        {
            if constexpr (Dimensions == 0)
            {
                return 1;
            }
            else
            {
                return ele_count_.size();
            }
        }

        size_type max_size() const noexcept
        {
            return size();
        }

        bool empty() const noexcept
        {
            return size() == 0;
        }

        iterator begin() const
        {
            return resolved_pointer();
        }

        iterator end() const
        {
            return resolved_pointer() + size();
        }

        const_iterator cbegin() const
        {
            return resolved_pointer();
        }

        const_iterator cend() const
        {
            return resolved_pointer() + size();
        }

        reverse_iterator rbegin() const
        {
            return reverse_iterator(end());
        }

        reverse_iterator rend() const
        {
            return reverse_iterator(begin());
        }

        const_reverse_iterator crbegin() const
        {
            return const_reverse_iterator(cend());
        }

        const_reverse_iterator crend() const
        {
            return const_reverse_iterator(cbegin());
        }

        template <
            int D = Dimensions,
            typename std::enable_if_t<(D > 0)> * = nullptr>
        range<Dimensions> get_range() const
        {
            return ele_count_;
        }

        PARAS_KERNEL_HD
        const local_accessor &operator=(const value_type &val) const
        {
            *resolved_pointer() = val;
            return *this;
        }

        template <
            int D = Dimensions,
            typename std::enable_if_t<D == 0> * = nullptr>
        PARAS_KERNEL_HD
        operator reference() const
        {
            return *resolved_pointer();
        }

        template <
            int D = Dimensions,
            typename std::enable_if_t<(D > 0), int> = 0>
        PARAS_KERNEL_HD
            reference
            operator[](const id<Dimensions> &index) const
        {
            return resolved_pointer()[linear_id(index)];
        }

        class row
        {
            mutable DataT *base_;

        public:
            PARAS_KERNEL_HD
            row(DataT *p) : base_(p) {}
            PARAS_KERNEL_HD
            DataT &operator[](size_t i) const { return base_[i]; }
        };

        class plane
        {
            mutable DataT *base_;
            size_t stride1_;

        public:
            PARAS_KERNEL_HD
            plane(DataT *p, size_t stride1) : base_(p), stride1_(stride1) {}
            PARAS_KERNEL_HD
            row operator[](size_t i) const { return row(base_ + i * stride1_); }
        };

        template <
            int D = Dimensions,
            typename std::enable_if_t<(D == 2), int> = 0>
        PARAS_KERNEL_HD
        row operator[](size_type index) const
        {
            return row(resolved_pointer() + index * ele_count_[1]);
        }

        template <
            int D = Dimensions,
            typename std::enable_if_t<(D == 3), int> = 0>
        PARAS_KERNEL_HD
        plane operator[](size_type index) const
        {
            return plane(resolved_pointer() + index * ele_count_[1] * ele_count_[2], ele_count_[2]);
        }

        template <
            int D = Dimensions,
            typename std::enable_if_t<(D == 1), int> = 0>
        PARAS_KERNEL_HD
            reference
            operator[](size_type index) const
        {
            return resolved_pointer()[index];
        }

        template <access::decorated IsDecorated>
        PARAS_KERNEL_HD
            accessor_ptr<IsDecorated>
            get_multi_ptr() const noexcept
        {
            return accessor_ptr<IsDecorated>{
                resolved_pointer()};
        }

        PARAS_KERNEL_HD
        local_ptr<value_type> get_pointer() const noexcept
        {
            return get_multi_ptr<access::decorated::legacy>();
        }

        PARAS_KERNEL_HD
        operator accessor_ptr<access::decorated::legacy>() const noexcept
        {
            return accessor_ptr<access::decorated::legacy>{
                resolved_pointer()};
        }

    private:

        size_type local_offset_bytes_ = 0;

        range<Dimensions> ele_count_{};

        DataT *host_storage_ = nullptr;
        property_list props_{};

        PARAS_KERNEL_HD
        DataT *resolved_pointer() const noexcept
        {
#if PARAS_DEVICE_PASS
            unsigned char *const base =
                paras_get_dynamic_shared_memory();

            return reinterpret_cast<DataT *>(
                base + local_offset_bytes_);
#else

            if (paras_host_detail::tl_local_mem)
                return reinterpret_cast<DataT *>(
                    paras_host_detail::tl_local_mem + local_offset_bytes_);

            return host_storage_;
#endif
        }

        PARAS_KERNEL_HD
        size_type linear_id(const id<Dimensions> &index) const noexcept
        {
            size_type linear = 0;

            for (int i = 0; i < Dimensions; ++i)
            {
                linear =
                    linear * static_cast<size_type>(
                                 ele_count_[i]) +
                    static_cast<size_type>(
                        index[i]);
            }

            return linear;
        }

        PARAS_KERNEL_HD
        friend bool operator==(const local_accessor &lhs, const local_accessor &rhs) noexcept
        {
            return lhs.host_storage_ == rhs.host_storage_ &&
                   lhs.local_offset_bytes_ == rhs.local_offset_bytes_;
        }
        PARAS_KERNEL_HD
        friend bool operator!=(const local_accessor &lhs, const local_accessor &rhs) noexcept
        {
            return !(lhs == rhs);
        }
    };

}

namespace sycl
{

#if !PARAS_NO_BUFFERS
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
multi_ptr(accessor<DataT, Dimensions, AccessMode, target::device, IsPlaceholder>)
    -> multi_ptr<std::conditional_t<AccessMode == access::mode::read, const DataT, DataT>,
                 access::address_space::global_space, access::decorated::no>;
#endif

template <typename DataT, int Dimensions>
multi_ptr(local_accessor<DataT, Dimensions>)
    -> multi_ptr<DataT, access::address_space::local_space, access::decorated::no>;

}

namespace std
{
#if !PARAS_NO_BUFFERS
    template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
              sycl::target AccessTarget, sycl::access::placeholder IsPlaceholder>
    struct hash<sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>>
    {
        size_t operator()(
            const sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder> &a) const noexcept
        {
            return std::hash<const void *>{}(static_cast<const void *>(a.paras_raw()));
        }
    };

    template <typename DataT, int Dimensions, sycl::access_mode AccessMode>
    struct hash<sycl::host_accessor<DataT, Dimensions, AccessMode>>
    {
        size_t operator()(
            const sycl::host_accessor<DataT, Dimensions, AccessMode> &a) const noexcept
        {
            return std::hash<const void *>{}(static_cast<const void *>(a.paras_raw()));
        }
    };

#endif

    template <typename DataT, int Dimensions>
    struct hash<sycl::local_accessor<DataT, Dimensions>>
    {
        size_t operator()(
            const sycl::local_accessor<DataT, Dimensions> &a) const noexcept
        {

            return std::hash<const void *>{}(static_cast<const void *>(a.begin()));
        }
    };
}

#pragma GCC diagnostic pop

#endif

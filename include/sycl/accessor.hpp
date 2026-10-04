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

    namespace access
    {
        using target = ::sycl::target;
    }

    template <
    typename DataT,
    int Dimensions = 1,
    access::mode AccessMode = access::mode::read_write,
    target AccessTarget = target::device,
    access::placeholder IsPlaceholder = access::placeholder::false_t>
    class accessor
    {
        mutable DataT *ptr_;
        range<Dimensions> r_;
        id<Dimensions> offset_{};
        bool no_init_ = false;

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
            : ptr_(const_cast<DataT *>(o.ptr_)), r_(o.r_), offset_(o.offset_), no_init_(o.no_init_) {}
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

        accessor(DataT *p, range<Dimensions> rr, access::mode)
            : ptr_(p), r_(rr) {}

        accessor() : ptr_(nullptr), r_() {}

        accessor(range<Dimensions> allocationSize, handler &,
                 const property_list &propList = {})
            : ptr_(nullptr), r_(allocationSize)
        {
            (void)propList;
        }

        template <access::decorated DecorateAddress>
        auto get_multi_ptr() const
        {
            if constexpr (AccessTarget == target::local)
            {
                return decorated_local_ptr<DataT>(ptr_);
            }
            else
            {
                return decorated_global_ptr<DataT>(ptr_);
            }
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
        }

        template <target T2 = AccessTarget, typename = std::enable_if_t<T2 == target::local>>
        accessor(range<Dimensions> allocationSize, handler &,
                 const property_list &props = {})
            : ptr_(new DataT[allocationSize.size()]()), r_(allocationSize)
        {
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 TagT, const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, TagT, const property_list &props = {}) : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange, id<Dimensions> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange, id<Dimensions> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange, TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 range<Dimensions> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, Dimensions, AllocatorT> &paras_buf,
                 handler &, range<Dimensions> accessRange, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        value_type *get_pointer() const { return ptr_; }

        DataT *begin() const { return ptr_; }
        DataT *end() const { return ptr_ + r_.size(); }

        const DataT *cbegin() const { return ptr_; }
        const DataT *cend() const { return ptr_ + r_.size(); }

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
        bool has_property() const noexcept
        {
            if constexpr (std::is_same_v<PropT, property::no_init>)
                return no_init_;
            else
                return false;
        }

        static constexpr bool is_placeholder() noexcept
        {
            return IsPlaceholder == access::placeholder::true_t;
        }

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
            return property::no_init{};
        }

        void swap(accessor &other) noexcept
        {
            using std::swap;
            swap(ptr_, other.ptr_);
            swap(r_, other.r_);
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
                return row(ptr_ + (offset_[0] + i) * r_[1] + offset_[1]);
            } else {
                return plane(ptr_ + (offset_[0] + i) * r_[1] * r_[2] + offset_[1] * r_[2] + offset_[2], r_[2]);
            }
        }

        reference operator[](id<Dimensions> idx) const
        {
            if constexpr (Dimensions == 1)
                return ptr_[offset_[0] + idx[0]];
            else if constexpr (Dimensions == 2)
                return ptr_[(offset_[0] + idx[0]) * r_[1] + (offset_[1] + idx[1])];
            else
                return ptr_[((offset_[0] + idx[0]) * r_[1] + (offset_[1] + idx[1])) * r_[2] + (offset_[2] + idx[2])];
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
        mutable DataT *ptr_;
        range<2> r_;
        id<2> offset_{};
        bool no_init_ = false;

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
            : ptr_(const_cast<DataT *>(o.ptr_)), r_(o.r_), offset_(o.offset_), no_init_(o.no_init_) {}
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

        accessor() : ptr_(nullptr), r_(range<2>(0, 0)) {}

        accessor(DataT *p, range<2> rr, access::mode)
            : ptr_(p), r_(rr) {}

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
        }

        template <target T2 = AccessTarget, typename = std::enable_if_t<T2 == target::local>>
        accessor(range<2> allocationSize, handler &,
                 const property_list &props = {})
            : ptr_(new DataT[allocationSize.size()]()), r_(allocationSize)
        {
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(paras_buf.get_range())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange, id<2> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange, id<2> accessOffset,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange, id<2> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange, TagT, const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange, id<2> accessOffset, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange), offset_(accessOffset)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange, accessOffset);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 range<2> accessRange,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 2, AllocatorT> &paras_buf,
                 handler &, range<2> accessRange, TagT,
                 const property_list &props = {})
            : ptr_(paras_buf.data()), r_(accessRange)
        {
            ::sycl::detail::wait_all_host_tasks();
            ::sycl::detail::check_accessor_range(paras_buf.get_range(), accessRange);
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        value_type *get_pointer() const { return ptr_; }

        template <access::decorated DecorateAddress>
        auto get_multi_ptr() const
        {
            return decorated_global_ptr<DataT>(ptr_);
        }

        DataT *begin() const { return ptr_; }
        DataT *end() const { return ptr_ + r_.size(); }

        const DataT *cbegin() const { return ptr_; }
        const DataT *cend() const { return ptr_ + r_.size(); }

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
        bool has_property() const noexcept
        {
            if constexpr (std::is_same_v<PropT, property::no_init>)
                return no_init_;
            else
                return false;
        }

        static constexpr bool is_placeholder() noexcept { return false; }

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
            return property::no_init{};
        }

        void swap(accessor &other) noexcept
        {
            using std::swap;
            swap(ptr_, other.ptr_);
            swap(r_, other.r_);
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
            return row(ptr_ + (offset_[0] + j) * r_[1] + offset_[1]);
        }

        reference operator[](id<2> idx) const
        {
            return ptr_[(offset_[0] + idx[0]) * r_[1] + (offset_[1] + idx[1])];
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
        mutable DataT *ptr_;
        bool no_init_ = false;

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
            : ptr_(const_cast<DataT *>(o.ptr_)), no_init_(o.no_init_) {}
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
            : ptr_(paras_buf.data())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT>
        accessor(buffer<DataT, 1, AllocatorT> &paras_buf,
                 handler &, const property_list &props = {})
            : ptr_(paras_buf.data())
        {
            ::sycl::detail::wait_all_host_tasks();
            no_init_ = props.has_no_init();
        }

        template <target T2 = AccessTarget, typename = std::enable_if_t<T2 == target::local>>
        accessor(handler &, const property_list &props = {})
            : ptr_(new DataT[1]())
        {
            no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 1, AllocatorT> &paras_buf,
                 TagT, const property_list &props = {})
            : ptr_(paras_buf.data())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        template <typename AllocatorT, typename TagT>
        accessor(buffer<DataT, 1, AllocatorT> &paras_buf,
                 handler &, TagT, const property_list &props = {})
            : ptr_(paras_buf.data())
        {
            ::sycl::detail::wait_all_host_tasks();
            if constexpr (std::is_same_v<TagT, property::no_init>)
                no_init_ = true;
            else
                no_init_ = props.has_no_init();
        }

        value_type *get_pointer() const { return ptr_; }

        template <access::decorated DecorateAddress>
        auto get_multi_ptr() const
        {
            return decorated_global_ptr<DataT>(ptr_);
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

        static constexpr bool is_placeholder() noexcept {
            return IsPlaceholder == access::placeholder::true_t;
        }

        size_t size() const noexcept { return 1; }

        bool empty() const noexcept { return false; }

        size_t byte_size() const noexcept { return sizeof(DataT); }

        size_t max_size() const noexcept { return 1; }

        DataT *begin() const { return ptr_; }
        DataT *end() const { return ptr_ + 1; }

        const DataT *cbegin() const { return ptr_; }
        const DataT *cend() const { return ptr_ + 1; }

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

        size_t get_count() const noexcept { return 1; }

        size_t get_size() const noexcept { return byte_size(); }
    };

    template <typename DataT, int Dimensions = 1, access::mode AccessMode = access::mode::read_write>
    using host_accessor = accessor<DataT, Dimensions, AccessMode, target::host_buffer>;

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

template <typename DataT, int Dimensions, access::mode AccessMode>
multi_ptr(accessor<DataT, Dimensions, AccessMode, target::device, access::placeholder::false_t>)
    -> multi_ptr<DataT, access::address_space::global_space, access::decorated::legacy>;

template <typename DataT, int Dimensions>
multi_ptr(local_accessor<DataT, Dimensions>)
    -> multi_ptr<DataT, access::address_space::local_space, access::decorated::legacy>;

}

namespace std
{
    template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
              sycl::target AccessTarget, sycl::access::placeholder IsPlaceholder>
    struct hash<sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>>
    {
        size_t operator()(
            const sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder> &a) const noexcept
        {
            return std::hash<const void *>{}(static_cast<const void *>(a.get_pointer()));
        }
    };

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

#endif

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
#include "id.hpp"
#include "access.hpp"
#include "property_list.hpp"
#include "handler.hpp"
#include "multi_ptr.hpp"
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

enum class target {
    device,
    host_task,
    host_buffer,
    global_buffer [[deprecated]] = device,
    constant_buffer [[deprecated]] = host_buffer + 1,
    local [[deprecated]],
    image [[deprecated]],
    host_image [[deprecated]],
    image_array [[deprecated]]
};

namespace access {
using target = ::sycl::target;
}

template <typename DataT, int Dimensions = 1, access::mode AccessMode = access::mode::read_write,
          target AccessTarget = target::device>
class accessor {
    DataT* base_ = nullptr;
    DataT* origin_ = nullptr; // base_ + offset_, precomputed for fast indexing
    range<Dimensions> buf_range_;
    range<Dimensions> acc_range_;
    id<Dimensions> offset_;
    bool no_init_ = false;

    template <typename TagT>
    static constexpr bool is_no_init_v = std::is_same_v<std::decay_t<TagT>, property::no_init>;

    PARAS_KERNEL_HD
    size_t rel(const id<Dimensions>& idx) const {
        if constexpr (Dimensions == 1)
            return idx[0];
        else if constexpr (Dimensions == 2)
            return idx[0] * buf_range_[1] + idx[1];
        else
            return (idx[0] * buf_range_[1] + idx[1]) * buf_range_[2] + idx[2];
    }

    void set_origin() {
        size_t lin = 0;
        for (int d = 0; d < Dimensions; ++d)
            lin = lin * buf_range_[d] + offset_[d];
        origin_ = base_ + lin;
    }

    template <typename AllocatorT>
    void bind(buffer<DataT, Dimensions, AllocatorT>& buf, const range<Dimensions>& r,
              const id<Dimensions>& off) {
        base_ = buf.data();
        buf_range_ = buf.get_range();
        acc_range_ = r;
        offset_ = off;
        set_origin();
        for (int d = 0; d < Dimensions; ++d) {
            if (off[d] + r[d] > buf_range_[d]) {
                throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                      "accessor range + offset exceeds the buffer range");
            }
        }
    }

public:
    using value_type = std::conditional_t<AccessMode == access::mode::read, const DataT, DataT>;
    using reference = value_type&;
    using const_reference = const DataT&;
    using size_type = size_t;
    template <access::decorated IsDecorated>
    using accessor_ptr = value_type*;

    accessor() = default;

    accessor(DataT* p, range<Dimensions> rr, access::mode)
        : base_(p), origin_(p), buf_range_(rr), acc_range_(rr) {}
    accessor(DataT* p, range<Dimensions> full, range<Dimensions> rr, id<Dimensions> off)
        : base_(p), buf_range_(full), acc_range_(rr), offset_(off) {
        set_origin();
    }

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, const property_list& props = {}) {
        bind(buf, buf.get_range(), id<Dimensions>());
        no_init_ = props.has_no_init();
    }
    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, handler&, const property_list& props = {})
        : accessor(buf, props) {}
    template <typename AllocatorT, typename TagT,
              typename = std::enable_if_t<!std::is_convertible_v<TagT, property_list> ||
                                          is_no_init_v<TagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, TagT, const property_list& props = {})
        : accessor(buf, props) {
        no_init_ = no_init_ || is_no_init_v<TagT>;
    }
    template <typename AllocatorT, typename TagT,
              typename = std::enable_if_t<!std::is_convertible_v<TagT, property_list> ||
                                          is_no_init_v<TagT>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, handler&, TagT,
             const property_list& props = {})
        : accessor(buf, props) {
        no_init_ = no_init_ || is_no_init_v<TagT>;
    }

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, range<Dimensions> r,
             const property_list& props = {}) {
        bind(buf, r, id<Dimensions>());
        no_init_ = props.has_no_init();
    }
    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, range<Dimensions> r, id<Dimensions> off,
             const property_list& props = {}) {
        bind(buf, r, off);
        no_init_ = props.has_no_init();
    }
    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, handler&, range<Dimensions> r,
             const property_list& props = {})
        : accessor(buf, r, props) {}
    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, handler&, range<Dimensions> r,
             id<Dimensions> off, const property_list& props = {})
        : accessor(buf, r, off, props) {}
    template <typename AllocatorT, typename TagT,
              typename = std::enable_if_t<!std::is_convertible_v<TagT, property_list>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, handler&, range<Dimensions> r, TagT,
             const property_list& props = {})
        : accessor(buf, r, props) {}
    template <typename AllocatorT, typename TagT,
              typename = std::enable_if_t<!std::is_convertible_v<TagT, property_list>>>
    accessor(buffer<DataT, Dimensions, AllocatorT>& buf, handler&, range<Dimensions> r,
             id<Dimensions> off, TagT, const property_list& props = {})
        : accessor(buf, r, off, props) {}

    range<Dimensions> get_range() const noexcept { return acc_range_; }
    id<Dimensions> get_offset() const noexcept { return offset_; }
    size_type size() const noexcept { return acc_range_.size(); }
    size_type byte_size() const noexcept { return size() * sizeof(DataT); }
    size_type get_count() const noexcept { return size(); }
    size_type get_size() const noexcept { return byte_size(); }
    size_type max_size() const noexcept { return size(); }
    bool empty() const noexcept { return size() == 0; }
    bool is_placeholder() const noexcept { return false; }

    template <typename PropT>
    bool has_property() const noexcept {
        if constexpr (std::is_same_v<PropT, property::no_init>)
            return no_init_;
        else
            return false;
    }

    PARAS_KERNEL_HD
    reference operator[](const id<Dimensions>& idx) const { return origin_[rel(idx)]; }

    template <int D = Dimensions, std::enable_if_t<D == 1, int> = 0>
    PARAS_KERNEL_HD
    reference operator[](size_t i) const {
        return origin_[i];
    }

    template <int D = Dimensions, std::enable_if_t<D == 2, int> = 0>
    PARAS_KERNEL_HD
    auto operator[](size_t i) const {
        return row{origin_ + i * buf_range_[1]};
    }
    template <int D = Dimensions, std::enable_if_t<D == 3, int> = 0>
    PARAS_KERNEL_HD
    auto operator[](size_t i) const {
        return plane{origin_ + i * buf_range_[1] * buf_range_[2], buf_range_[2]};
    }

    template <int D = Dimensions, std::enable_if_t<D == 0 || D == 1, int> = 0>
    PARAS_KERNEL_HD
    operator reference() const {
        return *origin_;
    }

    PARAS_KERNEL_HD
    DataT* get_pointer() const noexcept { return origin_; }

    template <access::decorated IsDecorated = access::decorated::no>
    PARAS_KERNEL_HD
    auto get_multi_ptr() const noexcept {
        return multi_ptr<value_type, access::address_space::global_space, IsDecorated>(
            get_pointer());
    }

    PARAS_KERNEL_HD
    value_type* begin() const noexcept { return get_pointer(); }

    PARAS_KERNEL_HD
    value_type* end() const noexcept { return get_pointer() + size(); }

    PARAS_KERNEL_HD
    const DataT* cbegin() const noexcept { return begin(); }

    PARAS_KERNEL_HD
    const DataT* cend() const noexcept { return end(); }

    template <typename CoordT>
    PARAS_KERNEL_HD
    DataT read(const CoordT& coords, const sampler&) const {
        return (*this)[coords];
    }

    template <typename CoordT>
    PARAS_KERNEL_HD
    void write(const CoordT& coords, const DataT& data) const {
        (*this)[coords] = data;
    }

private:
    struct row {
        value_type* p;

        PARAS_KERNEL_HD
        reference operator[](size_t k) const { return p[k]; }
    };

    struct plane {
        value_type* p;
        size_t stride;

        PARAS_KERNEL_HD
        row operator[](size_t j) const { return row{p + j * stride}; }
    };
};

template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, read_only_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, write_only_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::write, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, read_write_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, read_only_host_task_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read, target::host_task>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, write_only_host_task_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::write, target::host_task>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, read_write_host_task_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read_write, target::host_task>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, range<Dimensions>, read_only_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, range<Dimensions>, write_only_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::write, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, range<Dimensions>, read_write_tag_t,
         const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, range<Dimensions>, id<Dimensions>,
         read_only_tag_t, const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, range<Dimensions>, id<Dimensions>,
         write_only_tag_t, const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::write, target::device>;
template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>&, handler&, range<Dimensions>, id<Dimensions>,
         read_write_tag_t, const property_list& = {})
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device>;

template <typename DataT, int Dimensions = 1, access::mode AccessMode = access::mode::read_write>
using host_accessor = accessor<DataT, Dimensions, AccessMode, target::host_buffer>;

template <typename DataT, int Dimensions = 1>
class local_accessor {
public:
    using value_type = DataT;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = size_t;

    template <access::decorated IsDecorated>
    using accessor_ptr = multi_ptr<value_type, access::address_space::local_space, IsDecorated>;

    local_accessor() = default;

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

    PARAS_KERNEL_HD
    const local_accessor& operator=(const value_type& val) const {
        *resolved_pointer() = val;
        return *this;
    }

    template <int D = Dimensions, typename std::enable_if_t<D == 0>* = nullptr>
    PARAS_KERNEL_HD operator reference() const {
        return *resolved_pointer();
    }

    template <int D = Dimensions, typename std::enable_if_t<(D > 0), int> = 0>
    PARAS_KERNEL_HD reference operator[](const id<Dimensions>& index) const {
        return resolved_pointer()[linear_id(index)];
    }

    template <int D = Dimensions, typename std::enable_if_t<(D == 1), int> = 0>
    PARAS_KERNEL_HD reference operator[](size_type index) const {
        return resolved_pointer()[index];
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
        constexpr std::size_t kLocalMemoryScratchBytes = paras_host_detail::kHostLocalMemBytes;
        alignas(std::max_align_t) thread_local unsigned char
            local_memory_scratch[kLocalMemoryScratchBytes];
        return reinterpret_cast<DataT*>(local_memory_scratch + local_offset_bytes_);
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

template <typename DataT, int Dimensions, access::mode AccessMode>
class accessor<DataT, Dimensions, AccessMode, target::local>
    : public local_accessor<DataT, Dimensions> {
public:
    using local_accessor<DataT, Dimensions>::local_accessor;
};

} // namespace sycl

#endif

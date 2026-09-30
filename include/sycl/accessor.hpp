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

#include "../utilities/internal_utils.hpp"
#include "access.hpp"
#include "handler.hpp"
#include "id.hpp"
#include "local_ptr.hpp"
#include "multi_ptr.hpp"
#include "property_list.hpp"
#include "range.hpp"
#include <iterator>
#include <vector>

#include <cstddef>
#include <type_traits>
#include <utility>

#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename T, int Dimensions, typename AllocatorT> class buffer;

enum class target { device, host_buffer, host_task };

template <typename DataT, int Dimensions = 1,
          access::mode AccessMode = access::mode::read_write,
          target AccessTarget = target::device>
class accessor {
  mutable DataT *ptr_;
  range<Dimensions> r_;

public:
  using value_type = DataT;
  using reference = DataT &;

  template <access::decorated IsDecorated> using accessor_ptr = DataT *;

  accessor(DataT *p, range<Dimensions> rr, access::mode) : ptr_(p), r_(rr) {}

  accessor() : ptr_(nullptr), r_(range<Dimensions>(0)) {}

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &,
           const property_list & = {}) {}

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &,
           const property_list & = {}) {}

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &, TagT,
           const property_list & = {}) {}

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, TagT,
           const property_list & = {}) {}

  DataT *get_pointer() const { return ptr_; }

  DataT &operator[](size_t i) const { return ptr_[i]; }

  DataT &operator[](id<Dimensions> idx) const {
    if constexpr (Dimensions == 1)
      return ptr_[idx[0]];
    else
      return ptr_[idx[0] * r_[1] + idx[1]];
  }
};

template <typename DataT, access::mode AccessMode, target AccessTarget>
class accessor<DataT, 2, AccessMode, AccessTarget> {
  mutable DataT *ptr_;
  range<2> r_;

public:
  accessor() : ptr_(nullptr), r_(range<2>(0, 0)) {}

  accessor(DataT *p, range<2> rr, access::mode) : ptr_(p), r_(rr) {}

  template <typename AllocatorT>
  accessor(buffer<DataT, 2, AllocatorT> &, const property_list & = {}) {}

  DataT *get_pointer() const { return ptr_; }

  class row {
    mutable DataT *base_;

  public:
    row(DataT *p) : base_(p) {}
    DataT &operator[](size_t i) const { return base_[i]; }
  };

  row operator[](size_t j) const { return row(ptr_ + j * r_[1]); }

  DataT &operator[](id<2> idx) const { return ptr_[idx[0] * r_[1] + idx[1]]; }
};

template <typename DataT, int Dimensions = 1> class local_accessor {
public:
  using value_type = DataT;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = size_t;

  template <access::decorated IsDecorated>
  using accessor_ptr =
      multi_ptr<value_type, access::address_space::local_space, IsDecorated>;

  local_accessor() = default;

  template <int D = Dimensions, typename std::enable_if_t<D == 0> * = nullptr>
  explicit local_accessor(handler &cghRef, const property_list &propList = {})
      : local_offset_bytes_(cghRef.template local_alloc_offset<DataT>(1)),
        ele_count_{}, props_(propList) {}

  local_accessor(range<Dimensions> allocationRange, handler &cghRef,
                 const property_list &propList = {})
      : local_offset_bytes_(
            cghRef.template local_alloc_offset<DataT>(allocationRange.size())),
        ele_count_(allocationRange), props_(propList) {}

  void swap(local_accessor &other) {
    using std::swap;

    swap(local_offset_bytes_, other.local_offset_bytes_);
    swap(ele_count_, other.ele_count_);
    swap(props_, other.props_);
  }

  size_type byte_size() const noexcept { return size() * sizeof(DataT); }

  size_type size() const noexcept {
    if constexpr (Dimensions == 0) {
      return 1;
    } else {
      return ele_count_.size();
    }
  }

  bool empty() const noexcept { return size() == 0; }

  template <int D = Dimensions, typename std::enable_if_t<(D > 0)> * = nullptr>
  range<Dimensions> get_range() const {
    return ele_count_;
  }

  PARAS_KERNEL_HD
  const local_accessor &operator=(const value_type &val) const {
    *resolved_pointer() = val;
    return *this;
  }

  template <int D = Dimensions, typename std::enable_if_t<D == 0> * = nullptr>
  PARAS_KERNEL_HD operator reference() const {
    return *resolved_pointer();
  }

  template <int D = Dimensions, typename std::enable_if_t<(D > 0), int> = 0>
  PARAS_KERNEL_HD reference operator[](const id<Dimensions> &index) const {
    return resolved_pointer()[linear_id(index)];
  }

  template <int D = Dimensions, typename std::enable_if_t<(D == 1), int> = 0>
  PARAS_KERNEL_HD reference operator[](size_type index) const {
    return resolved_pointer()[index];
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
  DataT *resolved_pointer() const noexcept {
#if defined(__CUDA_ARCH__)
    unsigned char *const base = paras_get_dynamic_shared_memory();

    return reinterpret_cast<DataT *>(base + local_offset_bytes_);
#else
    return nullptr;
#endif
  }

  PARAS_KERNEL_HD
  size_type linear_id(const id<Dimensions> &index) const noexcept {
    size_type linear = 0;

    for (int i = 0; i < Dimensions; ++i) {
      linear = linear * static_cast<size_type>(ele_count_[i]) +
               static_cast<size_type>(index[i]);
    }

    return linear;
  }
};

} // namespace sycl

#endif

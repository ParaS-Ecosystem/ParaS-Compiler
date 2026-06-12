/**<
 * Copyright 2025 The ParaS-Compiler Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
  using const_reference = DataT &;
  template <access::decorated IsDecorated>
  using accessor_ptr =
      multi_ptr<value_type, access::address_space::local_space, IsDecorated>;
  using size_type = size_t;

  local_accessor() = default;

  template <int D = Dimensions, typename std::enable_if_t<D == 0> * = nullptr>
  explicit local_accessor(handler &cghRef, const property_list &propList = {})
      : local_ptr_{static_cast<DataT *>(cghRef.local_alloc<DataT>(1))},
        props_{propList} {}

  local_accessor(range<Dimensions> allocationRange, handler &cghRef,
                 const property_list &propList = {})
      : local_ptr_{static_cast<DataT *>(
            cghRef.local_alloc<DataT>(allocationRange.size()))},
        ele_count_{allocationRange}, props_{propList} {}

  void swap(local_accessor &other) {
    using std::swap;
    swap(local_ptr_, other.local_ptr_);
    swap(ele_count_, other.ele_count_);
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
    *local_ptr_ = val;
    return *this;
  }

  template <int D = Dimensions, typename std::enable_if_t<D == 0> * = nullptr>
  operator reference() const {
    return *local_ptr_;
  }

  template <int D = Dimensions, typename std::enable_if_t<(D > 1), int> = 0>
  reference operator[](size_type index) const {
    return local_ptr_[linear_id(index)];
  }

  template <int D = Dimensions, typename std::enable_if_t<(D == 1), int> = 0>
  PARAS_KERNEL_HD reference operator[](size_type index) const {
    return local_ptr_[index];
  }

  template <access::decorated IsDecorated>
  PARAS_KERNEL_HD accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
    return accessor_ptr<IsDecorated>{static_cast<DataT *>(local_ptr_)};
  }

  PARAS_KERNEL_HD
  local_ptr<value_type> get_pointer() const noexcept {
    return get_multi_ptr<access::decorated::legacy>();
  }

  PARAS_KERNEL_HD
  operator accessor_ptr<access::decorated::legacy>() const noexcept {
    return accessor_ptr<access::decorated::legacy>{
        static_cast<DataT *>(local_ptr_)};
  }

private:
  DataT *local_ptr_{};
  range<Dimensions> ele_count_{};
  range<Dimensions> ele_cnt_{};
  property_list props_;

  size_type linear_id(const id<Dimensions> &idnx) {
    size_type linear = 0;
    for (int i = 0; i < Dimensions; i++) {
      linear = linear * ele_cnt_[i] + idnx[i];
    }
    return linear;
  }
};

} // namespace sycl

#endif

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

#ifndef __PARAS_ND_ITEM_HPP__
#define __PARAS_ND_ITEM_HPP__

#include "group.hpp"
#include "id.hpp"
#include "nd_range.hpp"
#include "range.hpp"
#include "sub_group.hpp"

#define PARAS_KERNEL_D
#define PARAS_KERNEL_HD

namespace sycl {

template <int Dimensions = 1> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

private:
  id<Dimensions> gid_;
  id<Dimensions> lid_;
  id<Dimensions> grp_;

  range<Dimensions> global_range_;
  range<Dimensions> local_range_;
  range<Dimensions> group_range_;

public:
  PARAS_KERNEL_HD
  nd_item(id<Dimensions> global_id, id<Dimensions> local_id,
          id<Dimensions> group_id, range<Dimensions> global_r,
          range<Dimensions> local_r, range<Dimensions> group_r)
      : gid_(global_id), lid_(local_id), grp_(group_id),
        global_range_(global_r), local_range_(local_r), group_range_(group_r) {}

  PARAS_KERNEL_HD
  id<Dimensions> get_global_id() const { return gid_; }

  PARAS_KERNEL_HD
  size_t get_global_id(int dim) const { return gid_[dim]; }

  PARAS_KERNEL_HD
  size_t get_global_linear_id() const {
    size_t mul = 1, val = 0;
    for (int d = Dimensions - 1; d >= 0; d--) {
      val += gid_[d] * mul;
      mul *= global_range_[d];
    }
    return val;
  }

  PARAS_KERNEL_HD
  id<Dimensions> get_local_id() const { return lid_; }

  PARAS_KERNEL_HD
  size_t get_local_id(int dim) const { return lid_[dim]; }

  PARAS_KERNEL_HD
  size_t get_local_linear_id() const {
    size_t mul = 1, val = 0;
    for (int d = Dimensions - 1; d >= 0; d--) {
      val += lid_[d] * mul;
      mul *= local_range_[d];
    }
    return val;
  }

  PARAS_KERNEL_HD
  group<Dimensions> get_group() const {
    return group<Dimensions>(grp_, group_range_, local_range_);
  }

  PARAS_KERNEL_HD
  size_t get_group(int dim) const { return grp_[dim]; }

  PARAS_KERNEL_HD
  size_t get_group_linear_id() const {
    size_t mul = 1, val = 0;
    for (int d = Dimensions - 1; d >= 0; d--) {
      val += grp_[d] * mul;
      mul *= group_range_[d];
    }
    return val;
  }

  PARAS_KERNEL_HD
  range<Dimensions> get_group_range() const { return group_range_; }

  PARAS_KERNEL_HD
  range<Dimensions> get_global_range() const { return global_range_; }

  PARAS_KERNEL_HD
  range<Dimensions> get_local_range() const { return local_range_; }

  PARAS_KERNEL_HD
  size_t get_global_range(int d) const { return global_range_[d]; }

  PARAS_KERNEL_HD
  size_t get_local_range(int d) const { return local_range_[d]; }

  PARAS_KERNEL_HD
  size_t get_group_range(int d) const { return group_range_[d]; }

  PARAS_KERNEL_HD
  id<Dimensions> get_offset() const { return id<Dimensions>(); }

  PARAS_KERNEL_HD
  nd_range<Dimensions> get_nd_range() const {
    return nd_range<Dimensions>(global_range_, local_range_);
  }

  PARAS_KERNEL_HD
  sycl::sub_group get_sub_group() const { return sycl::sub_group{}; }

  PARAS_KERNEL_HD
  void barrier(access::fence_space) const { paras_syncthreads(); }

  PARAS_KERNEL_HD
  void barrier(memory_scope) const { paras_syncthreads(); }
};

} // namespace sycl

#endif

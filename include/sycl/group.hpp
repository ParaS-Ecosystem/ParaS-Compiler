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

#ifndef __PARAS_GROUP_HPP__
#define __PARAS_GROUP_HPP__

#include "id.hpp"
#include "range.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <int Dimensions = 1> class group {
public:
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = size_t;

  static constexpr int dimensions = Dimensions;

private:
  id_type group_id_;
  range_type group_range_;
  range_type local_range_;

public:
  PARAS_KERNEL_HD
  group(id_type gid, range_type gr, range_type lr)
      : group_id_(gid), group_range_(gr), local_range_(lr) {}

  PARAS_KERNEL_HD
  id_type get_group_id() const { return group_id_; }
  PARAS_KERNEL_HD
  size_t get_group_id(int dim) const { return group_id_[dim]; }

  PARAS_KERNEL_HD
  id_type get_local_id() const { return id_type(); }
  PARAS_KERNEL_HD
  size_t get_local_id(int) const { return 0; }

  PARAS_KERNEL_HD
  range_type get_local_range() const { return local_range_; }
  PARAS_KERNEL_HD
  size_t get_local_range(int dim) const { return local_range_[dim]; }

  PARAS_KERNEL_HD
  range_type get_group_range() const { return group_range_; }
  PARAS_KERNEL_HD
  size_t get_group_range(int dim) const { return group_range_[dim]; }

  PARAS_KERNEL_HD
  range_type get_max_local_range() const { return local_range_; }

  PARAS_KERNEL_HD
  size_t operator[](int dim) const { return group_id_[dim]; }

  PARAS_KERNEL_HD
  size_t get_group_linear_id() const {
    size_t linear = 0, mul = 1;
    for (int d = Dimensions - 1; d >= 0; d--) {
      linear += group_id_[d] * mul;
      mul *= group_range_[d];
    }
    return linear;
  }

  PARAS_KERNEL_HD
  size_t get_local_linear_id() const { return 0; }

  PARAS_KERNEL_HD
  size_t get_group_linear_range() const {
    size_t r = 1;
    for (int d = 0; d < Dimensions; d++)
      r *= group_range_[d];
    return r;
  }

  PARAS_KERNEL_HD
  size_t get_local_linear_range() const {
    size_t r = 1;
    for (int d = 0; d < Dimensions; d++)
      r *= local_range_[d];
    return r;
  }

  bool leader() const { return true; }

  template <typename Fn> void parallel_for_work_item(const Fn &fn) const {
    fn();
  }

  template <typename Fn>
  void parallel_for_work_item(range<Dimensions>, const Fn &fn) const {
    fn();
  }
};

} // namespace sycl

#endif

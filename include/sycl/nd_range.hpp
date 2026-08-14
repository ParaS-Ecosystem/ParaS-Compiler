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

#ifndef __PARAS_ND_RANGE_HPP__
#define __PARAS_ND_RANGE_HPP__

#include "id.hpp"
#include "range.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <int Dimensions = 1> class nd_range {
public:
  static constexpr int dimensions = Dimensions;

private:
  range<Dimensions> global_range_;
  range<Dimensions> local_range_;
  id<Dimensions> offset_;

public:
  PARAS_KERNEL_HD
  nd_range(range<Dimensions> g, range<Dimensions> l,
           id<Dimensions> o = id<Dimensions>())
      : global_range_(g), local_range_(l), offset_(o) {}

  PARAS_KERNEL_HD
  range<Dimensions> get_global_range() const { return global_range_; }

  PARAS_KERNEL_HD
  range<Dimensions> get_local_range() const { return local_range_; }

  PARAS_KERNEL_HD
  range<Dimensions> get_group_range() const {
    range<Dimensions> gr;
    for (int d = 0; d < Dimensions; d++)
      gr[d] = global_range_[d] / local_range_[d];
    return gr;
  }

  PARAS_KERNEL_HD
  id<Dimensions> get_offset() const { return offset_; }
};

} // namespace sycl

#endif

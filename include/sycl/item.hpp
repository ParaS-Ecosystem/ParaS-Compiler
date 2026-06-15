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

#ifndef __PARAS_ITEM_HPP__
#define __PARAS_ITEM_HPP__

#include "id.hpp"

#define PARAS_KERNEL_D
#define PARAS_KERNEL_HD

namespace sycl {

template <int Dims> class item;

template <> class item<2> {
  id<2> idx_;

public:
  PARAS_KERNEL_HD
  item(const id<2> &id) : idx_(id) {}

  PARAS_KERNEL_HD
  item(size_t i, size_t j) : idx_(id<2>(i, j)) {}

  PARAS_KERNEL_HD
  size_t &operator[](int dim) { return idx_[dim]; }

  PARAS_KERNEL_HD
  size_t operator[](int dim) const { return idx_[dim]; }

  PARAS_KERNEL_HD
  id<2> get_id() const { return idx_; }
};

} // namespace sycl

#endif

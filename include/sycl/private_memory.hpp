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

#ifndef __PARAS_PRIVATE_MEMORY_HPP__
#define __PARAS_PRIVATE_MEMORY_HPP__

#include <cstddef>

#include "group.hpp"
#include "h_item.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <typename T, int Dimensions = 1> class private_memory {
public:
  PARAS_KERNEL_HD
  explicit private_memory(const group<Dimensions> &g)
      : range_(g.get_local_range()) {
    std::size_t n = 1;
    for (int d = 0; d < Dimensions; ++d)
      n *= range_[d];
    storage_ = new T[n];
  }
  private_memory(const private_memory &) = delete;
  private_memory &operator=(const private_memory &) = delete;
  PARAS_KERNEL_HD ~private_memory() { delete[] storage_; }

  PARAS_KERNEL_HD
  T &operator()(const h_item<Dimensions> &item) {
    const id<Dimensions> lid = item.get_local_id();
    std::size_t lin = 0;
    for (int d = 0; d < Dimensions; ++d)
      lin = lin * range_[d] + lid[d];
    return storage_[lin];
  }

private:
  range<Dimensions> range_;
  T *storage_ = nullptr;
};

}

#endif

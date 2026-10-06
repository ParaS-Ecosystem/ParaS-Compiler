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

#ifndef __PARAS_H_ITEM_HPP__
#define __PARAS_H_ITEM_HPP__

#include "id.hpp"
#include "range.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <int Dimensions = 1>
class h_item {
    id<Dimensions> global_id_;
    range<Dimensions> global_range_;
    id<Dimensions> local_id_;
    range<Dimensions> local_range_;

public:
    h_item() = delete;

    PARAS_KERNEL_HD
    h_item(id<Dimensions> globalId, range<Dimensions> globalRange, id<Dimensions> localId,
           range<Dimensions> localRange)
        : global_id_(globalId), global_range_(globalRange), local_id_(localId),
          local_range_(localRange) {}

    PARAS_KERNEL_HD
    id<Dimensions> get_global_id() const { return global_id_; }
    PARAS_KERNEL_HD
    std::size_t get_global_id(int dimension) const { return global_id_[dimension]; }
    PARAS_KERNEL_HD
    range<Dimensions> get_global_range() const { return global_range_; }
    PARAS_KERNEL_HD
    std::size_t get_global_range(int dimension) const { return global_range_[dimension]; }

    PARAS_KERNEL_HD
    id<Dimensions> get_local_id() const { return local_id_; }
    PARAS_KERNEL_HD
    std::size_t get_local_id(int dimension) const { return local_id_[dimension]; }
    PARAS_KERNEL_HD
    range<Dimensions> get_local_range() const { return local_range_; }
    PARAS_KERNEL_HD
    std::size_t get_local_range(int dimension) const { return local_range_[dimension]; }

    PARAS_KERNEL_HD
    id<Dimensions> get_logical_local_id() const { return local_id_; }
    PARAS_KERNEL_HD
    id<Dimensions> get_physical_local_id() const { return local_id_; }
    PARAS_KERNEL_HD
    range<Dimensions> get_logical_local_range() const { return local_range_; }
    PARAS_KERNEL_HD
    range<Dimensions> get_physical_local_range() const { return local_range_; }
};

} // namespace sycl

#endif

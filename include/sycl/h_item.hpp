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
#include "item.hpp"
#include "kem_gpu/gpu_utilities.hpp"

namespace sycl {

template <int Dimensions = 1>
class h_item {
    id<Dimensions> global_id_;
    range<Dimensions> global_range_;
    id<Dimensions> local_id_;
    range<Dimensions> local_range_;
    id<Dimensions> physical_id_;
    range<Dimensions> physical_range_;

public:
    h_item() = delete;

    PARAS_KERNEL_HD
    h_item(id<Dimensions> globalId, range<Dimensions> globalRange, id<Dimensions> localId,
           range<Dimensions> localRange)
        : global_id_(globalId), global_range_(globalRange), local_id_(localId),
          local_range_(localRange), physical_id_(localId), physical_range_(localRange) {}

    PARAS_KERNEL_HD
    h_item(id<Dimensions> globalId, range<Dimensions> globalRange, id<Dimensions> logicalId,
           range<Dimensions> logicalRange, id<Dimensions> physicalId,
           range<Dimensions> physicalRange)
        : global_id_(globalId), global_range_(globalRange), local_id_(logicalId),
          local_range_(logicalRange), physical_id_(physicalId), physical_range_(physicalRange) {}

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
    id<Dimensions> get_physical_local_id() const { return physical_id_; }
    PARAS_KERNEL_HD
    range<Dimensions> get_logical_local_range() const { return local_range_; }
    PARAS_KERNEL_HD
    range<Dimensions> get_physical_local_range() const { return physical_range_; }

    static constexpr int dimensions = Dimensions;
    PARAS_KERNEL_HD
    item<Dimensions, false> get_global() const {
        return item<Dimensions, false>(global_id_, global_range_);
    }
    PARAS_KERNEL_HD
    item<Dimensions, false> get_local() const { return get_logical_local(); }
    PARAS_KERNEL_HD
    item<Dimensions, false> get_logical_local() const {
        return item<Dimensions, false>(local_id_, local_range_);
    }
    PARAS_KERNEL_HD
    item<Dimensions, false> get_physical_local() const {
        return item<Dimensions, false>(physical_id_, physical_range_);
    }
    PARAS_KERNEL_HD
    std::size_t get_logical_local_id(int dimension) const { return local_id_[dimension]; }
    PARAS_KERNEL_HD
    std::size_t get_physical_local_id(int dimension) const { return physical_id_[dimension]; }
    PARAS_KERNEL_HD
    std::size_t get_logical_local_range(int dimension) const { return local_range_[dimension]; }
    PARAS_KERNEL_HD
    std::size_t get_physical_local_range(int dimension) const {
        return physical_range_[dimension];
    }
    PARAS_KERNEL_HD
    friend bool operator==(const h_item& a, const h_item& b) {
        return a.global_id_ == b.global_id_ && a.global_range_ == b.global_range_ &&
               a.local_id_ == b.local_id_ && a.local_range_ == b.local_range_ &&
               a.physical_id_ == b.physical_id_ && a.physical_range_ == b.physical_range_;
    }
    PARAS_KERNEL_HD
    friend bool operator!=(const h_item& a, const h_item& b) { return !(a == b); }
};

} // namespace sycl

#endif

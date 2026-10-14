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
#include <vector>
#include "group.hpp"
#include "h_item.hpp"

namespace sycl {

template <typename T, int Dimensions = 1>
class [[deprecated("sycl::private_memory is deprecated in SYCL 2020")]] private_memory {
public:
    private_memory(const group<Dimensions>& g) : data_(g.get_local_range().size()) {}

    T& operator()(const h_item<Dimensions>& id) {
        return data_[id.get_physical_local().get_linear_id()];
    }

private:
    std::vector<T> data_;
};

}

#endif

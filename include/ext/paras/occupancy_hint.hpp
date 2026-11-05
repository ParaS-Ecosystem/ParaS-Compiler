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

#ifndef __SYCL_EXT_PARAS_OCCUPANCY_HINT_HPP__
#define __SYCL_EXT_PARAS_OCCUPANCY_HINT_HPP__

#include <type_traits>

namespace sycl {
namespace ext {
namespace paras {
namespace detail {

template <typename F, typename = void>
struct occupancy_hint : std::false_type {};

template <typename F>
struct occupancy_hint<F, std::void_t<decltype(F::paras_max_work_group_size),
                                     decltype(F::paras_min_waves_per_eu)>> : std::true_type {
    static constexpr int max_work_group_size = F::paras_max_work_group_size;
    static constexpr int min_waves_per_eu = F::paras_min_waves_per_eu;
};

}
}
}
}

#endif

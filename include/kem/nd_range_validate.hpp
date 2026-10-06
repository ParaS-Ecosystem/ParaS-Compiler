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

#ifndef __PARAS_ND_RANGE_VALIDATE_HPP__
#define __PARAS_ND_RANGE_VALIDATE_HPP__

#include <string>
#include "sycl/exception.hpp"
#include "sycl/nd_range.hpp"

namespace paras_detail {

template <int D>
inline void validate_nd_range(const sycl::nd_range<D>& r) {
    const auto G = r.get_global_range();
    const auto L = r.get_local_range();
    for (int d = 0; d < D; ++d) {
        if (L[d] == 0 || G[d] % L[d] != 0) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                                  "nd_range: global size " + std::to_string(G[d]) +
                                      " is not divisible by local size " + std::to_string(L[d]) +
                                      " in dimension " + std::to_string(d));
        }
    }
}

} // namespace paras_detail

#endif

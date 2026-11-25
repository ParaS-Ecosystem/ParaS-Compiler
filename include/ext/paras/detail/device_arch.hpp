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

#ifndef __SYCL_EXT_PARAS_DEVICE_ARCH_HPP__
#define __SYCL_EXT_PARAS_DEVICE_ARCH_HPP__

#include <cctype>
#include <cstddef>
#include <string>

namespace sycl {
namespace ext {
namespace paras {
namespace detail {

inline bool is_four_digit_gfx(const std::string& version) {
    const auto p = version.find("gfx");
    if (p == std::string::npos) return false;
    std::size_t digits = 0;
    while (p + 3 + digits < version.size() &&
           std::isxdigit(static_cast<unsigned char>(version[p + 3 + digits])))
        ++digits;
    return digits >= 4;
}

}
}
}
}

#endif

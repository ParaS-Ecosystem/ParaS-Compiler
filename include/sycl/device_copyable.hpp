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

#ifndef __PARAS_DEVICE_COPYABLE_HPP__
#define __PARAS_DEVICE_COPYABLE_HPP__

#include <type_traits>

namespace sycl {

template <typename T>
struct is_device_copyable : std::bool_constant<std::is_trivially_copyable_v<T>> {};

template <typename T>
inline constexpr bool is_device_copyable_v = is_device_copyable<T>::value;

} // namespace sycl

#define SYCL_ADD_DEVICE_COPYABLE(T)                                                                \
    namespace sycl {                                                                               \
    template <>                                                                                    \
    struct is_device_copyable<T> : std::true_type {};                                              \
    }

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

#endif

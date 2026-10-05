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

#ifndef __PARAS_DECORATED_PTR_DEVICE_EVENT_HPP__
#define __PARAS_DECORATED_PTR_DEVICE_EVENT_HPP__

#include "multi_ptr.hpp"

namespace sycl {

template <typename T>
using decorated_local_ptr =
    multi_ptr<T, access::address_space::local_space, access::decorated::yes>;
template <typename T>
using decorated_global_ptr =
    multi_ptr<T, access::address_space::global_space, access::decorated::yes>;
template <typename T>
using decorated_private_ptr =
    multi_ptr<T, access::address_space::private_space, access::decorated::yes>;
template <typename T>
using decorated_generic_ptr =
    multi_ptr<T, access::address_space::generic_space, access::decorated::yes>;

class device_event {
public:
    constexpr device_event() noexcept = default;
    PARAS_KERNEL_HD void wait() const {}
};

} // namespace sycl

#endif

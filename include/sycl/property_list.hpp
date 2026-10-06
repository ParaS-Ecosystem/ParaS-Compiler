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

#ifndef __PARAS_PROPERTY_LIST_HPP__
#define __PARAS_PROPERTY_LIST_HPP__

#include <utility>
#include <type_traits>

namespace sycl {

namespace property {

namespace queue {

struct in_order {};
struct enable_profiling {};
} // namespace queue

struct no_init {};

namespace reduction {
struct initialize_to_identity {};
} // namespace reduction
} // namespace property

inline constexpr property::no_init no_init{};

namespace detail {
template <typename T>
struct is_property : std::false_type {};
template <>
struct is_property<property::queue::in_order> : std::true_type {};
template <>
struct is_property<property::queue::enable_profiling> : std::true_type {};
template <>
struct is_property<property::no_init> : std::true_type {};
template <>
struct is_property<property::reduction::initialize_to_identity> : std::true_type {};
} // namespace detail

class property_list {

private:
    bool in_order_ = false;
    bool profiling_ = false;
    bool no_init_ = false;
    bool initialize_to_identity_ = false;

    void store(property::queue::in_order) { in_order_ = true; }

    void store(property::queue::enable_profiling) { profiling_ = true; }

    void store(property::no_init) { no_init_ = true; }

    void store(property::reduction::initialize_to_identity) { initialize_to_identity_ = true; }

public:
    property_list() noexcept = default;

    template <
        typename... Properties,
        typename = std::enable_if_t<(detail::is_property<std::decay_t<Properties>>::value && ...)>>
    property_list(Properties&&... props) : property_list() {
        (store(std::forward<Properties>(props)), ...);
    }

    bool has_in_order() const { return in_order_; }
    bool has_profiling() const { return profiling_; }
    bool has_no_init() const { return no_init_; }
    bool has_initialize_to_identity() const { return initialize_to_identity_; }
};

} // namespace sycl

#endif

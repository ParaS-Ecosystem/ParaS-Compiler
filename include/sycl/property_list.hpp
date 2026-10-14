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
#include <memory>
#include <mutex>
#include <vector>

namespace sycl {

class context;

namespace property {

namespace queue {

struct in_order {};
struct enable_profiling {};
} // namespace queue

struct no_init {};

namespace reduction {
struct initialize_to_identity {};
} // namespace reduction

namespace detail {
class paras_context_holder {
public:
    paras_context_holder() = default;
    explicit paras_context_holder(std::shared_ptr<const void> c) : ctx_(std::move(c)) {}
    const std::shared_ptr<const void>& get() const noexcept { return ctx_; }

private:
    std::shared_ptr<const void> ctx_;
};
} // namespace detail

namespace buffer {
struct use_host_ptr {};
class use_mutex {
public:
    use_mutex(std::mutex& m) : mutex_(&m) {}
    std::mutex* get_mutex_ptr() const noexcept { return mutex_; }

private:
    std::mutex* mutex_;
};
class context_bound {
public:
    context_bound(sycl::context boundContext);
    sycl::context get_context() const;

private:
    detail::paras_context_holder holder_;
};
}

namespace image {
struct use_host_ptr {};
class use_mutex {
public:
    use_mutex(std::mutex& m) : mutex_(&m) {}
    std::mutex* get_mutex_ptr() const noexcept { return mutex_; }

private:
    std::mutex* mutex_;
};
class context_bound {
public:
    context_bound(sycl::context boundContext);
    sycl::context get_context() const;

private:
    detail::paras_context_holder holder_;
};
}
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
template <>
struct is_property<property::buffer::use_host_ptr> : std::true_type {};
template <>
struct is_property<property::buffer::use_mutex> : std::true_type {};
template <>
struct is_property<property::buffer::context_bound> : std::true_type {};
template <>
struct is_property<property::image::use_host_ptr> : std::true_type {};
template <>
struct is_property<property::image::use_mutex> : std::true_type {};
template <>
struct is_property<property::image::context_bound> : std::true_type {};

template <typename T>
struct paras_prop_key {
    static inline const char id = 0;
};
[[noreturn]] inline void paras_throw_missing_property();
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

    std::vector<std::pair<const void*, std::shared_ptr<const void>>> paras_props_;
    template <typename P>
    void paras_keep(const P& p) {
        paras_props_.emplace_back(&detail::paras_prop_key<P>::id, std::make_shared<P>(p));
    }
    template <typename P>
    void store(const P&) {}

public:
    property_list() noexcept = default;

    template <
        typename... Properties,
        typename = std::enable_if_t<(detail::is_property<std::decay_t<Properties>>::value && ...)>>
    property_list(Properties&&... props) : property_list() {
        (store(std::forward<Properties>(props)), ...);
        (paras_keep(static_cast<const std::decay_t<Properties>&>(props)), ...);
    }

    template <typename P>
    bool paras_has_property() const noexcept {
        for (const auto& kv : paras_props_)
            if (kv.first == &detail::paras_prop_key<P>::id)
                return true;
        return false;
    }
    template <typename P>
    P paras_get_property() const {
        for (const auto& kv : paras_props_)
            if (kv.first == &detail::paras_prop_key<P>::id)
                return *static_cast<const P*>(kv.second.get());
        detail::paras_throw_missing_property();
    }

    bool has_in_order() const { return in_order_; }
    bool has_profiling() const { return profiling_; }
    bool has_no_init() const { return no_init_; }
    bool has_initialize_to_identity() const { return initialize_to_identity_; }
};

template <typename Property>
struct is_property : detail::is_property<Property> {};
template <typename Property>
inline constexpr bool is_property_v = is_property<Property>::value;
template <typename Property, typename SyclObject>
struct is_property_of : std::false_type {};
template <typename Property, typename SyclObject>
inline constexpr bool is_property_of_v = is_property_of<Property, SyclObject>::value;

} // namespace sycl

#endif

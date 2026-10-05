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

#ifndef __PARAS_CONTEXT_HPP__
#define __PARAS_CONTEXT_HPP__

#include <vector>
#include <memory>
#include "sycl/property_list.hpp"
#include "sycl/device.hpp"
#include <functional>
#include "sycl/exception.hpp"
#include "sycl/atomic_ref.hpp"

namespace sycl {

using async_handler = std::function<void(sycl::exception_list)>;

namespace info {
namespace context {
struct platform {
    using return_type = sycl::platform;
};
struct devices {
    using return_type = std::vector<sycl::device>;
};
struct atomic_memory_order_capabilities {
    using return_type = std::vector<memory_order>;
};
struct atomic_memory_scope_capabilities {
    using return_type = std::vector<memory_scope>;
};
struct atomic_fence_order_capabilities {
    using return_type = std::vector<memory_order>;
};
struct atomic_fence_scope_capabilities {
    using return_type = std::vector<memory_scope>;
};
} // namespace context
} // namespace info

class context {
public:
    explicit context(const property_list& = {}) : devices_{device{}} {}

    explicit context(async_handler, const property_list& = {}) : devices_{device{}} {}

    explicit context(const device& dev, const property_list& = {}) : devices_{dev} {}

    explicit context(const device& dev, async_handler, const property_list& = {}) : devices_{dev} {}

    explicit context(const std::vector<device>& deviceList, const property_list& = {})
        : devices_{deviceList.empty()
                       ? throw sycl::exception(sycl::make_error_code(sycl::errc::invalid))
                       : deviceList} {}

    explicit context(const std::vector<device>& deviceList, async_handler,
                     const property_list& = {})
        : devices_{deviceList.empty()
                       ? throw sycl::exception(sycl::make_error_code(sycl::errc::invalid))
                       : deviceList} {}

    explicit context(const platform& plt, const property_list& = {})
        : devices_{plt.get_devices()} {}

    explicit context(const platform& plt, async_handler, const property_list& = {})
        : devices_{plt.get_devices()} {}

    backend get_backend() const noexcept {
        return devices_.empty() ? backend::host : devices_.front().get_backend();
    }

    platform get_platform() const {
        return devices_.empty() ? platform("host") : devices_.front().get_platform();
    }

    std::vector<device> get_devices() const { return devices_; }

    const device& get_device() const {
        static device fallback{};
        return devices_.empty() ? fallback : devices_.front();
    }

    template <typename Param>
    typename Param::return_type get_info() const {
        return typename Param::return_type{};
    }

    template <typename Param>
    typename Param::return_type get_backend_info() const {
        return typename Param::return_type{};
    }

    bool operator==(const context& rhs) const noexcept { return id_ == rhs.id_; }

    bool operator!=(const context& rhs) const noexcept { return !(*this == rhs); }

    std::shared_ptr<int> get_id_() const noexcept { return id_; }

private:
    std::vector<device> devices_;
    std::shared_ptr<int> id_{std::make_shared<int>()};
};

inline exception::exception(const context& ctx, std::error_code ec, const std::string& what_arg)
    : exception(ec, what_arg) {
    context_ = std::make_shared<context>(ctx);
}

inline exception::exception(const context& ctx, std::error_code ec)
    : exception(ctx, ec, ec.message()) {}

inline exception::exception(const context& ctx, int ev, const std::error_category& cat,
                            const std::string& what_arg)
    : exception(ctx, std::error_code(ev, cat), what_arg) {}

inline exception::exception(const context& ctx, int ev, const std::error_category& cat)
    : exception(ctx, std::error_code(ev, cat)) {}

inline context exception::get_context() const {
    return *context_;
}

template <>
inline typename info::context::platform::return_type
context::get_info<info::context::platform>() const {
    return get_platform();
}

template <>
inline typename info::context::devices::return_type
context::get_info<info::context::devices>() const {
    return get_devices();
}

template <>
inline typename info::context::atomic_memory_order_capabilities::return_type
context::get_info<info::context::atomic_memory_order_capabilities>() const {
    return {memory_order::relaxed, memory_order::acquire, memory_order::release,
            memory_order::acq_rel, memory_order::seq_cst};
}

template <>
inline typename info::context::atomic_fence_order_capabilities::return_type
context::get_info<info::context::atomic_fence_order_capabilities>() const {
    return {memory_order::relaxed, memory_order::acquire, memory_order::release,
            memory_order::acq_rel, memory_order::seq_cst};
}

template <>
inline typename info::context::atomic_memory_scope_capabilities::return_type
context::get_info<info::context::atomic_memory_scope_capabilities>() const {
    return {memory_scope::work_item, memory_scope::sub_group, memory_scope::work_group};
}

template <>
inline typename info::context::atomic_fence_scope_capabilities::return_type
context::get_info<info::context::atomic_fence_scope_capabilities>() const {
    return {memory_scope::work_item, memory_scope::sub_group, memory_scope::work_group};
}

} // namespace sycl

namespace std {
template <>
struct hash<sycl::context> {
    std::size_t operator()(const sycl::context& c) const noexcept {
        return std::hash<std::shared_ptr<int>>{}(c.get_id_());
    }
};
} // namespace std

#endif

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

#ifndef __PARAS_KERNEL_HPP__
#define __PARAS_KERNEL_HPP__

#include <vector>
#include <string>
#include <type_traits>

#include "context.hpp"
#include "kernel_handler.hpp"

namespace sycl
{

    enum class bundle_state
    {
        input,
        object,
        executable
    };

    class kernel
    {
    public:
        kernel() = delete;

        sycl::context get_context() const { return context_; }

        sycl::backend get_backend() const noexcept { return backend::host; }

        template <typename Param>
        typename Param::return_type get_info() const
        {

            if constexpr (std::is_same_v<Param, info::kernel::num_args>)
                throw exception(make_error_code(errc::invalid),
                                "info::kernel::num_args needs an interop or "
                                "built-in kernel");
            return typename Param::return_type{};
        }

        template <typename Param>
        typename Param::return_type get_info(const device &dev) const
        {
            namespace kds = info::kernel_device_specific;
            if constexpr (std::is_same_v<Param, kds::global_work_size>)
            {

                throw exception(make_error_code(errc::invalid),
                                "global_work_size needs a custom device or a "
                                "built-in kernel");
            }
            else if constexpr (std::is_same_v<Param, kds::work_group_size>)
            {
                return dev.get_info<info::device::max_work_group_size>();
            }
            else if constexpr (std::is_same_v<Param, kds::preferred_work_group_size_multiple>)
            {
                return 1;
            }
            else if constexpr (std::is_same_v<Param, kds::max_num_sub_groups>)
            {
                const std::uint32_t n = dev.get_info<info::device::max_num_sub_groups>();
                return n ? n : 1;
            }
            else if constexpr (std::is_same_v<Param, kds::max_sub_group_size>)
            {
                std::uint32_t m = 0;
                for (auto s : dev.get_info<info::device::sub_group_sizes>())
                    if (static_cast<std::uint32_t>(s) > m)
                        m = static_cast<std::uint32_t>(s);
                return m ? m : 1;
            }
            else
            {
                return typename Param::return_type{};
            }
        }

        bool operator==(const kernel &rhs) const { return context_ == rhs.context_; }
        bool operator!=(const kernel &rhs) const { return !(*this == rhs); }

    private:
        template <bundle_state>
        friend class kernel_bundle;

        explicit kernel(sycl::context ctx) : context_(ctx) {}

        sycl::context context_;
    };

    class kernel_id
    {
    public:
        kernel_id() = default;

        bool operator==(const kernel_id &) const noexcept { return true; }
        bool operator!=(const kernel_id &) const noexcept { return false; }

        const char *get_name() const noexcept { return "paras_kernel"; }
    };

    template <typename KernelName>
    inline kernel_id get_kernel_id()
    {
        return kernel_id{};
    }

    inline std::vector<kernel_id> get_kernel_ids()
    {
        return {};
    }

    inline bool is_compatible(const std::vector<kernel_id> &, const device &)
    {
        return true;
    }

    template <bundle_state State>
    class device_image
    {
    public:
        device_image() = default;

        bool operator==(const device_image &) const noexcept { return true; }
        bool operator!=(const device_image &) const noexcept { return false; }

        bool has_kernel(const kernel_id &) const noexcept { return true; }

        bool has_kernel(const kernel_id &, const device &) const noexcept { return true; }
        template <typename KernelName>
        bool has_kernel() const noexcept { return true; }
    };

    template <bundle_state State>
    class kernel_bundle
    {
    public:
        kernel get_kernel(const kernel_id &) const { return kernel(sycl::context()); }

        template <typename KernelName>
        kernel get_kernel() const { return kernel(sycl::context()); }

        bool empty() const noexcept { return true; }

        using iterator = const device_image<State> *;
        using const_iterator = const device_image<State> *;
        iterator begin() const noexcept { return nullptr; }
        iterator end() const noexcept { return nullptr; }

        bool has_kernel(const kernel_id &) const noexcept
        {
            return true;
        }

        bool has_kernel(const kernel_id &, const device &) const noexcept
        {
            return true;
        }

        template <typename KernelName>
        bool has_kernel() const noexcept { return true; }

        template <typename KernelName>
        bool has_kernel(const device &) const noexcept { return true; }

        std::vector<kernel_id> get_kernel_ids() const { return {}; }

        sycl::context get_context() const { return sycl::context{}; }

        using device_image_iterator = const device_image<State> *;

        bool operator==(const kernel_bundle &) const noexcept { return true; }
        bool operator!=(const kernel_bundle &) const noexcept { return false; }

        sycl::backend get_backend() const noexcept { return backend::host; }

        template <auto &SpecName>
        bool has_specialization_constant() const noexcept { return false; }

        template <auto &SpecName>
        typename std::remove_reference_t<decltype(SpecName)>::value_type
        get_specialization_constant() const
        {

            return detail::spec_const_value_host<SpecName>();
        }

        bool contains_specialization_constants() const noexcept { return false; }

        template <typename T>
        const T *native_specialization_constant(const std::string &) const noexcept
        {
            return nullptr;
        }

        void *native_specialization_constant() const noexcept { return nullptr; }

        template <auto &SpecName>
        void set_specialization_constant(
            typename std::remove_reference_t<decltype(SpecName)>::value_type value)
        {
            detail::spec_const_set_host<SpecName>(value);
        }

        std::vector<device> get_devices() const { return {}; }
    };

    template <bundle_state State>
    inline kernel_bundle<State> join(const std::vector<kernel_bundle<State>> &)
    {
        return kernel_bundle<State>{};
    }

}

namespace sycl
{

    template <bundle_state State>
    inline kernel_bundle<State> get_kernel_bundle(const context &)
    {
        return kernel_bundle<State>{};
    }

    template <bundle_state State>
    inline kernel_bundle<State> get_kernel_bundle(const context &,
                                                  const std::vector<device> &)
    {
        return kernel_bundle<State>{};
    }

    template <bundle_state State>
    inline kernel_bundle<State> get_kernel_bundle(const context &,
                                                  const std::vector<kernel_id> &)
    {
        return kernel_bundle<State>{};
    }

    template <typename KernelName, bundle_state State>
    inline kernel_bundle<State> get_kernel_bundle(const context &)
    {
        return kernel_bundle<State>{};
    }

    template <typename KernelName, bundle_state State>
    inline kernel_bundle<State> get_kernel_bundle(const context &,
                                                  const std::vector<device> &)
    {
        return kernel_bundle<State>{};
    }

    template <bundle_state State, typename Selector>
    inline kernel_bundle<State> get_kernel_bundle(const context &, Selector)
    {
        return kernel_bundle<State>{};
    }

    template <bundle_state State>
    inline kernel_bundle<State> get_kernel_bundle(const context &,
                                                  const std::vector<device> &,
                                                  const std::vector<kernel_id> &)
    {
        return kernel_bundle<State>{};
    }

    template <bundle_state State, typename Selector>
    inline kernel_bundle<State> get_kernel_bundle(const context &,
                                                  const std::vector<device> &,
                                                  Selector)
    {
        return kernel_bundle<State>{};
    }

    inline kernel_bundle<bundle_state::executable> link(
        const std::vector<kernel_bundle<bundle_state::object>> &,
        const std::vector<device> & = {},
        const property_list & = {})
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    inline kernel_bundle<bundle_state::executable> link(
        const kernel_bundle<bundle_state::object> &,
        const std::vector<device> & = {},
        const property_list & = {})
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    inline kernel_bundle<bundle_state::executable> link(
        const std::vector<kernel_bundle<bundle_state::object>> &,
        const property_list &)
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    inline kernel_bundle<bundle_state::executable> link(
        const kernel_bundle<bundle_state::object> &,
        const property_list &)
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    namespace detail {
    template <bundle_state State> constexpr bool paras_has_state()
    {
        return State == bundle_state::executable;
    }
    }

    template <bundle_state State>
    inline bool has_kernel_bundle(const context &)
    {
        return detail::paras_has_state<State>();
    }

    template <bundle_state State>
    inline bool has_kernel_bundle(const context &, const std::vector<kernel_id> &)
    {
        return detail::paras_has_state<State>();
    }

    template <bundle_state State>
    inline bool has_kernel_bundle(const context &, const std::vector<device> &)
    {
        return detail::paras_has_state<State>();
    }

    template <bundle_state State>
    inline bool has_kernel_bundle(const context &, const std::vector<device> &,
                                  const std::vector<kernel_id> &)
    {
        return detail::paras_has_state<State>();
    }

    template <typename KernelName, bundle_state State>
    inline bool has_kernel_bundle(const context &)
    {
        return detail::paras_has_state<State>();
    }

    template <typename KernelName, bundle_state State>
    inline bool has_kernel_bundle(const context &, const std::vector<device> &)
    {
        return detail::paras_has_state<State>();
    }

    inline kernel_bundle<bundle_state::executable>
    build(const kernel_bundle<bundle_state::input> &)
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    inline kernel_bundle<bundle_state::executable>
    build(const kernel_bundle<bundle_state::input> &,
          const std::vector<device> &)
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    inline kernel_bundle<bundle_state::executable>
    build(const kernel_bundle<bundle_state::input> &,
          const std::vector<device> &,
          const property_list &)
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    inline kernel_bundle<bundle_state::executable>
    build(const kernel_bundle<bundle_state::input> &,
          const property_list &)
    {
        return kernel_bundle<bundle_state::executable>{};
    }

    inline kernel_bundle<bundle_state::object> compile(
        const kernel_bundle<bundle_state::input> &,
        const std::vector<device> & = {})
    {
        return kernel_bundle<bundle_state::object>{};
    }

    inline kernel_bundle<bundle_state::object> compile(
        const kernel_bundle<bundle_state::input> &,
        const property_list &)
    {
        return kernel_bundle<bundle_state::object>{};
    }

    inline kernel_bundle<bundle_state::object> compile(
        const kernel_bundle<bundle_state::input> &,
        const std::vector<device> &,
        const property_list &)
    {
        return kernel_bundle<bundle_state::object>{};
    }

    template <>
    inline auto device::get_info<info::device::built_in_kernel_ids>() const
    {
        return std::vector<kernel_id>{};
    }

}

namespace std
{
    template <>
    struct hash<sycl::kernel>
    {
        std::size_t operator()(const sycl::kernel &k) const noexcept
        {
            return std::hash<sycl::context>{}(k.get_context());
        }
    };

    template <>
    struct hash<sycl::kernel_id>
    {
        std::size_t operator()(const sycl::kernel_id &) const noexcept { return 0; }
    };

    template <sycl::bundle_state State>
    struct hash<sycl::device_image<State>>
    {
        std::size_t operator()(const sycl::device_image<State> &) const noexcept { return 0; }
    };

    template <sycl::bundle_state State>
    struct hash<sycl::kernel_bundle<State>>
    {
        std::size_t operator()(const sycl::kernel_bundle<State> &) const noexcept { return 0; }
    };
}

#endif

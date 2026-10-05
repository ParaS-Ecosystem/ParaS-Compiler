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

#ifndef __PARAS_EXCEPTION_EXCEPTION_HPP__
#define __PARAS_EXCEPTION_EXCEPTION_HPP__

#include <stdexcept>
#include <string>
#include <vector>
#include <exception>
#include <system_error>
#include <memory>
namespace sycl {

enum class errc {
    success = 0,
    runtime,
    kernel,
    accessor,
    nd_range,
    event,
    kernel_argument,
    build,
    invalid,
    memory_allocation,
    platform,
    profiling,
    feature_not_supported,
    kernel_not_supported,
    backend_mismatch
};

} // namespace sycl

namespace std {
template <>
struct is_error_code_enum<sycl::errc> : public std::true_type {};
} // namespace std

namespace sycl {
class context;
class sycl_category_impl : public std::error_category {
public:
    const char* name() const noexcept override { return "sycl"; }
    std::string message(int) const override { return "sycl error"; }
};

inline const std::error_category& sycl_category() noexcept {
    static sycl_category_impl instance;
    return instance;
}

inline std::error_code make_error_code(errc e) noexcept {
    return std::error_code(static_cast<int>(e), sycl_category());
}

class exception : public std::runtime_error {
public:
    explicit exception(const std::string& message)
        : std::runtime_error(message), code_(make_error_code(errc::success)) {}

    exception(const std::string& message, const std::string& backend_info)
        : std::runtime_error(message), backend_info_(backend_info),
          code_(make_error_code(errc::success)) {}

    exception(std::error_code ec, const std::string& what_arg)
        : std::runtime_error(what_arg), code_(ec) {}

    explicit exception(std::error_code ec) : std::runtime_error(ec.message()), code_(ec) {}

    exception(int ev, const std::error_category& cat, const std::string& what_arg)
        : std::runtime_error(what_arg), code_(ev, cat) {}

    exception(int ev, const std::error_category& cat)
        : std::runtime_error(cat.message(ev)), code_(ev, cat) {}

    exception(const context& ctx, std::error_code ec, const std::string& what_arg);

    explicit exception(const context& ctx, std::error_code ec);
    exception(const context& ctx, int ev, const std::error_category& cat,
              const std::string& what_arg);
    exception(const context& ctx, int ev, const std::error_category& cat);

    [[nodiscard]]
    const std::string& backend_info() const noexcept {
        return backend_info_;
    }

    [[nodiscard]]
    std::error_code code() const noexcept {
        return code_;
    }

    [[nodiscard]]
    const std::error_category& category() const noexcept {
        return code_.category();
    }
    bool has_context() const noexcept { return static_cast<bool>(context_); }

    context get_context() const;

private:
    std::string backend_info_;
    std::error_code code_;
    std::shared_ptr<context> context_;
};

class exception_list {
public:
    using value_type = std::exception_ptr;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;
    using iterator = std::vector<std::exception_ptr>::iterator;
    using const_iterator = std::vector<std::exception_ptr>::const_iterator;

    size_type size() const noexcept { return list_.size(); }
    iterator begin() noexcept { return list_.begin(); }
    iterator end() noexcept { return list_.end(); }
    const_iterator begin() const noexcept { return list_.begin(); }
    const_iterator end() const noexcept { return list_.end(); }

private:
    std::vector<std::exception_ptr> list_;
};

} // namespace sycl

#endif

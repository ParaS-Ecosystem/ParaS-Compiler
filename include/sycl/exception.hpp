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

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

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

}

namespace std {
template <> struct is_error_code_enum<sycl::errc> : public std::true_type {};
}

namespace sycl {

namespace detail {
class sycl_category_impl : public std::error_category {
public:
  const char *name() const noexcept override { return "sycl"; }
  std::string message(int ev) const override {
    switch (static_cast<errc>(ev)) {
    case errc::success: return "success";
    case errc::runtime: return "runtime error";
    case errc::kernel: return "kernel error";
    case errc::accessor: return "accessor error";
    case errc::nd_range: return "invalid nd_range";
    case errc::event: return "event error";
    case errc::kernel_argument: return "invalid kernel argument";
    case errc::build: return "build error";
    case errc::invalid: return "invalid";
    case errc::memory_allocation: return "memory allocation failed";
    case errc::platform: return "platform error";
    case errc::profiling: return "profiling error";
    case errc::feature_not_supported: return "feature not supported";
    case errc::kernel_not_supported: return "kernel not supported";
    case errc::backend_mismatch: return "backend mismatch";
    }
    return "unknown sycl error";
  }
};
}

inline const std::error_category &sycl_category() noexcept {
  static detail::sycl_category_impl instance;
  return instance;
}

inline std::error_code make_error_code(errc e) noexcept {
  return std::error_code(static_cast<int>(e), sycl_category());
}

class exception : public std::runtime_error {
public:
  explicit exception(const std::string &message)
      : std::runtime_error(message), code_(make_error_code(errc::runtime)) {}

  exception(const std::string &message, const std::string &backend_info)
      : std::runtime_error(message), backend_info_(backend_info),
        code_(make_error_code(errc::runtime)) {}

  exception(std::error_code ec, const char *what_arg)
      : std::runtime_error(what_arg), code_(ec) {}
  exception(std::error_code ec, const std::string &what_arg)
      : std::runtime_error(what_arg), code_(ec) {}
  explicit exception(std::error_code ec)
      : std::runtime_error(ec.message()), code_(ec) {}
  exception(int ev, const std::error_category &cat, const char *what_arg)
      : std::runtime_error(what_arg), code_(ev, cat) {}
  exception(int ev, const std::error_category &cat,
            const std::string &what_arg)
      : std::runtime_error(what_arg), code_(ev, cat) {}
  exception(int ev, const std::error_category &cat)
      : std::runtime_error(cat.message(ev)), code_(ev, cat) {}

  std::string backend_info() const noexcept { return backend_info_; }

  const std::error_code &code() const noexcept { return code_; }
  const std::error_category &category() const noexcept {
    return code_.category();
  }

private:
  std::string backend_info_;
  std::error_code code_;
};

class exception_list {
public:
  using value_type = std::exception_ptr;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = std::size_t;
  using iterator = std::vector<std::exception_ptr>::const_iterator;
  using const_iterator = std::vector<std::exception_ptr>::const_iterator;

  size_type size() const noexcept { return list_.size(); }
  iterator begin() const noexcept { return list_.begin(); }
  iterator end() const noexcept { return list_.end(); }

private:
  std::vector<std::exception_ptr> list_;
};

}

#endif

/**<
 * Copyright 2025 The ParaS-Compiler Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_EXCEPTION_EXCEPTION_HPP__
#define __PARAS_EXCEPTION_EXCEPTION_HPP__

#include <stdexcept>
#include <string>

namespace sycl {

class exception : public std::runtime_error {
public:
  explicit exception(const std::string &message)
      : std::runtime_error(message) {}

  exception(const std::string &message, const std::string &backend_info)
      : std::runtime_error(message), backend_info_(backend_info) {}

  std::string backend_info() const noexcept { return backend_info_; }

private:
  std::string backend_info_;
};

} // namespace sycl

#endif

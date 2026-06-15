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

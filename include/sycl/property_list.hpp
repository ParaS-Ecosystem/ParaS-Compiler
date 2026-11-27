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

namespace sycl {

namespace property {

namespace queue {

struct in_order {};
struct enable_profiling {};
} // namespace queue
} // namespace property

class property_list {

private:
  bool in_order_ = false;
  bool profiling_ = false;

  void store(property::queue::in_order) { in_order_ = true; }

  void store(property::queue::enable_profiling) { profiling_ = true; }

public:
  property_list() noexcept = default;

  template <typename... Properties>
  property_list(Properties &&...props) : property_list() {
    (store(std::forward<Properties>(props)), ...);
  }

  bool has_in_order() const { return in_order_; }
  bool has_profiling() const { return profiling_; }
};

} // namespace sycl

#endif

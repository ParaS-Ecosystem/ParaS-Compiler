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

#ifndef __PARAS_INFO_HPP__
#define __PARAS_INFO_HPP__

namespace sycl {
namespace info {

enum class event_command_status : unsigned int { submitted, running, complete };

namespace event {

struct command_execution_status {
  using return_type = sycl::info::event_command_status;
};

} // namespace event

namespace event_profiling {

struct command_submit {
  using return_type = unsigned long long;
};

struct command_start {
  using return_type = unsigned long long;
};

struct command_end {
  using return_type = unsigned long long;
};

} // namespace event_profiling
} // namespace info
} // namespace sycl

#endif

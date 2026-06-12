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

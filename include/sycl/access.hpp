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

#ifndef __PARAS_ACCESS_HPP__
#define __PARAS_ACCESS_HPP__

namespace sycl {
namespace access {

enum class mode { read, write, read_write, discard_write, discard_read_write };

enum class address_space {
  private_space,
  global_space,
  generic_space,
  local_space,
  constant_space,
};

enum class decorated {
  no,
  yes,
  legacy,
  read_only,
  write_only,
  read_write,
};

enum class fence_space : char { local_space, global_space, global_and_local };

} // namespace access

using access_mode = access::mode;

} // namespace sycl

#endif

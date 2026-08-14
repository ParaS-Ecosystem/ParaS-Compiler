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

#ifndef __PARAS_INTEROP_HANDLE_HPP__
#define __PARAS_INTEROP_HANDLE_HPP__

#include "device.hpp"

struct CUstream_st;

namespace sycl {

class queue;
class handler;

class interop_handle {
private:
  void *backend_ptr_;
  backend backend_;

  interop_handle(void *p, backend b) : backend_ptr_(p), backend_(b) {}

  friend class handler;

public:
  interop_handle() = delete;

  backend get_backend() const noexcept { return backend_; }

  void *get_native_queue();

  template <backend Backend>
  backend_return_t<Backend, queue> get_native_queue() const;
};

} // namespace sycl

#endif

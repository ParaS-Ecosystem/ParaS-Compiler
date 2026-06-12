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

#ifndef __PARAS_INTEROP_HANDLER_HPP__
#define __PARAS_INTEROP_HANDLER_HPP__

namespace sycl {

class interop_handle {
  void *backend_ptr_;

public:
  interop_handle(void *p) : backend_ptr_(p) {}

  void *get_native_queue();
};

} // namespace sycl

#endif

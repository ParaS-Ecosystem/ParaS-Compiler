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

#ifndef __PARAS_THREADPOOL_EXECUTE_COMMON_HPP__
#define __PARAS_THREADPOOL_EXECUTE_COMMON_HPP__

#include <thread>

inline unsigned threadpool::get_num_threads() {
  unsigned n = std::thread::hardware_concurrency();
  return (n == 0 ? 4 : n);
}
#endif

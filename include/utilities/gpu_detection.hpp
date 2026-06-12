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

#ifndef __PARAS_GPU_DETECTION_HPP__
#define __PARAS_GPU_DETECTION_HPP__

#include "../sycl/device.hpp"
#include <vector>

namespace paras_extension {
std::vector<sycl::device> detect_nvidia_gpus();
std::vector<sycl::device> detect_amd_gpus();
std::vector<sycl::device> detect_all_gpus();
} // namespace paras_extension

#endif

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

#ifndef __PARAS_HD_REGION_HPP__
#define __PARAS_HD_REGION_HPP__

#if defined(__HIP__) || defined(__CUDA__)
#include "kem_gpu/hd_region_std.hpp"
#define PARAS_HD_PRAGMA_(x) _Pragma(#x)
#define PARAS_HD_REGION_OPEN                                                   \
  PARAS_HD_PRAGMA_(clang force_cuda_host_device begin)
#define PARAS_HD_REGION_CLOSE                                                  \
  PARAS_HD_PRAGMA_(clang force_cuda_host_device end)
#define PARAS_HD_REGION_PAUSE                                                  \
  PARAS_HD_PRAGMA_(clang force_cuda_host_device end)
#define PARAS_HD_REGION_RESUME                                                 \
  PARAS_HD_PRAGMA_(clang force_cuda_host_device begin)
#else
#define PARAS_HD_REGION_OPEN
#define PARAS_HD_REGION_CLOSE
#define PARAS_HD_REGION_PAUSE
#define PARAS_HD_REGION_RESUME
#endif

#endif

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

#ifndef __PARAS_GPU_THREADPOOL_EXECUTE_COMMON_HPP__
#define __PARAS_GPU_THREADPOOL_EXECUTE_COMMON_HPP__

#include <cuda_runtime.h>
#include <thread>

using std::cout;

inline unsigned cuda_threadpool::gpu_get_num_threads() {

  unsigned int devCount = 0, totalThreads;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devCount);

  int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
  int numsSM = prop.multiProcessorCount;

  totalThreads = maxThreadsPerSM * numsSM;

  return (totalThreads == 0 ? 32 : totalThreads);
}

#endif /** End of gpu_threadpool_execute_common >*/

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

#ifndef __PARAS_GPU_HANDEL_IMPL_HPP__
#define __PARAS_GPU_HANDEL_IMPL_HPP__

#include "kem_gpu/gpu_threadpool.hpp"
#include "sycl/handler.hpp"

namespace sycl {

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(range<dim> r, Func f) {
    if constexpr (std::is_invocable_v<const Func&, id<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](id<dim> i) { f(i, kernel_handler{}); });
        return;
    } else if constexpr (std::is_invocable_v<const Func&, item<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](item<dim> i) { f(i, kernel_handler{}); });
        return;
    } else {
#if (PARAS_CUDA_BACKEND)
        if (gpu_pool_ == nullptr) {
            throw std::runtime_error("GPU handler has no cuda_threadpool backend");
        }

        wait_for_dependencies();

        if constexpr (dim == 1) {
            if (isAsyncEnabled()) {
                gpu_pool_->gpu_execute_1D_async(f);
            } else {
                gpu_pool_->gpu_execute_1D(r, f);
            }
        } else if constexpr (dim == 2) {
            gpu_pool_->gpu_execute_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_pool_->gpu_execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
#elif (PARAS_HIP_BACKEND)
        if (rocm_pool_ == nullptr) {
            throw std::runtime_error("GPU handler has no rocm_threadpool backend");
        }

        wait_for_dependencies();

        if constexpr (dim == 1) {
            if (isAsyncEnabled()) {
                rocm_pool_->gpu_execute_1D_async(f);
            } else {
                rocm_pool_->gpu_execute_1D(r, f);
            }
        } else if constexpr (dim == 2) {
            rocm_pool_->gpu_execute_2D(r, f);
        } else if constexpr (dim == 3) {
            rocm_pool_->gpu_execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
#endif
    }
}

template <typename KernelName, typename Func, int dim>
void handler::parallel_for(const nd_range<dim>& r, Func f) {
    if constexpr (std::is_invocable_v<const Func&, nd_item<dim>, kernel_handler>) {
        this->template parallel_for<KernelName>(r, [f](nd_item<dim> i) { f(i, kernel_handler{}); });
        return;
    } else {
#if (PARAS_CUDA_BACKEND)
        if (gpu_pool_ == nullptr) {
            throw std::runtime_error("GPU handler has no cuda_threadpool backend");
        }

        wait_for_dependencies();

        const std::size_t sharedMemoryBytes_cuda = local_memory_size();

        if constexpr (dim == 1) {
            gpu_pool_->gpu_execute_nd_range_1D(r, f, sharedMemoryBytes_cuda);
        } else if constexpr (dim == 2) {
            gpu_pool_->gpu_execute_nd_range_2D(r, f, sharedMemoryBytes_cuda);
        } else if constexpr (dim == 3) {
            gpu_pool_->gpu_execute_nd_range_3D(r, f, sharedMemoryBytes_cuda);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }

#elif (PARAS_HIP_BACKEND)
        if (rocm_pool_ == nullptr) {
            throw std::runtime_error("GPU handler has no rocm_threadpool backend");
        }

        wait_for_dependencies();

        const std::size_t sharedMemoryBytes_hip = local_memory_size();

        if constexpr (dim == 1) {
            rocm_pool_->gpu_execute_nd_range_1D(r, f, sharedMemoryBytes_hip);
        } else if constexpr (dim == 2) {
            rocm_pool_->gpu_execute_nd_range_2D(r, f, sharedMemoryBytes_hip);
        } else if constexpr (dim == 3) {
            rocm_pool_->gpu_execute_nd_range_3D(r, f, sharedMemoryBytes_hip);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
#endif
    }
}

} // namespace sycl

#endif /** End of handler_impl >*/

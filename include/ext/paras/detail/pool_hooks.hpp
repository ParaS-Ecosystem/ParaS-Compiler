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

#ifndef __SYCL_EXT_PARAS_POOL_HOOKS_HPP__
#define __SYCL_EXT_PARAS_POOL_HOOKS_HPP__

namespace sycl {

inline void handler::wait_for_prior_commands() {
    wait_for_dependencies();
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_ != nullptr && gpu_pool_->get_device().is_gpu()) gpu_pool_->wait();
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ != nullptr && rocm_pool_->get_device().is_gpu()) rocm_pool_->wait();
#endif
}

inline int handler::paras_activate_native_device() {
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_ == nullptr || !gpu_pool_->get_device().is_gpu()) return -1;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
    const cudaError_t err = cudaSetDevice(gpu_pool_->get_device().get_native_id());
    if (err != cudaSuccess) throw sycl::exception(cudaGetErrorString(err));
    return previous;
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ == nullptr || !rocm_pool_->get_device().is_gpu()) return -1;
    int previous = -1;
    if (hipGetDevice(&previous) != hipSuccess) previous = -1;
    const hipError_t err = hipSetDevice(rocm_pool_->get_device().get_native_id());
    if (err != hipSuccess) throw sycl::exception(hipGetErrorString(err));
    return previous;
#else
    return -1;
#endif
}

inline void handler::paras_restore_native_device(int previous) noexcept {
    if (previous < 0) return;
#if (PARAS_CUDA_BACKEND)
    (void)cudaSetDevice(previous);
#elif (PARAS_HIP_BACKEND)
    (void)hipSetDevice(previous);
#endif
}

#if (PARAS_CUDA_BACKEND)
#define PARAS_HANDLER_POOL gpu_pool_
#elif (PARAS_HIP_BACKEND)
#define PARAS_HANDLER_POOL rocm_pool_
#endif

inline bool handler::paras_recording() const {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    return PARAS_HANDLER_POOL != nullptr && PARAS_HANDLER_POOL->is_recording();
#else
    return false;
#endif
}

inline void handler::paras_record_host_task(std::function<void()> fn) {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    wait_for_dependencies();
    PARAS_HANDLER_POOL->paras_record_host_task(std::move(fn));
#else
    fn();
#endif
}

inline void handler::paras_fail_recording(const char* what) const {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    if (PARAS_HANDLER_POOL != nullptr) PARAS_HANDLER_POOL->paras_fail_recording(what);
#endif
    throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                          std::string(what) + " cannot be recorded into a graph");
}

inline void handler::paras_prefetch(const void* ptr, std::size_t num_bytes) {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    if (PARAS_HANDLER_POOL == nullptr || !PARAS_HANDLER_POOL->get_device().is_gpu()) return;
    wait_for_dependencies();
    (void)PARAS_HANDLER_POOL->prefetch(ptr, num_bytes);
#else
    (void)ptr;
    (void)num_bytes;
#endif
}

inline void handler::paras_mem_advise(const void* ptr, std::size_t num_bytes, int advice) {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    if (PARAS_HANDLER_POOL == nullptr || !PARAS_HANDLER_POOL->get_device().is_gpu()) return;
    wait_for_dependencies();
    (void)PARAS_HANDLER_POOL->mem_advise(ptr, num_bytes, advice);
#else
    (void)ptr;
    (void)num_bytes;
    (void)advice;
#endif
}
#undef PARAS_HANDLER_POOL

}

#if (PARAS_CUDA_BACKEND)
inline cuda_threadpool& sycl::queue::paras_ext_pool() const { return get_or_create_gpu_pool(); }
#elif (PARAS_HIP_BACKEND)
inline rocm_threadpool& sycl::queue::paras_ext_pool() const { return get_or_create_rocm_pool(); }
#endif

inline sycl::event sycl::queue::paras_submit_event() const {
    auto& pool = paras_ext_pool();
    if (pool.async_launch() || !pool.paras_profiling()) return pool.paras_finish_event(0);
    return event{get_backend()};
}


#endif

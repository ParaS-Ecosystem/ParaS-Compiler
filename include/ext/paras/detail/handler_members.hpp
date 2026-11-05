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

    void wait_for_prior_commands();
    bool paras_recording() const;
    void paras_record_host_task(std::function<void()> fn);
    [[noreturn]] void paras_fail_recording(const char* what) const;

    int paras_activate_native_device();
    static void paras_restore_native_device(int previous) noexcept;

    struct native_device_scope {
        explicit native_device_scope(handler& h) : previous(h.paras_activate_native_device()) {}
        ~native_device_scope() { paras_restore_native_device(previous); }
        native_device_scope(const native_device_scope&) = delete;
        native_device_scope& operator=(const native_device_scope&) = delete;
        int previous;
    };

    void mem_advise(const void* ptr, std::size_t num_bytes, int advice) {
        paras_mem_advise(ptr, num_bytes, advice);
    }
    void prefetch(const void* ptr, std::size_t num_bytes) { paras_prefetch(ptr, num_bytes); }
    inline void paras_prefetch(const void* ptr, std::size_t num_bytes);
    inline void paras_mem_advise(const void* ptr, std::size_t num_bytes, int advice);

    interop_handle host_task_interop_handle() {
#if (PARAS_CUDA_BACKEND)
        if (gpu_pool_ != nullptr) {
            wait_for_dependencies();
            return interop_handle(static_cast<void*>(gpu_pool_), sycl::backend::cuda);
        }
#elif (PARAS_HIP_BACKEND)
        if (rocm_pool_ != nullptr) {
            wait_for_dependencies();
            return interop_handle(static_cast<void*>(rocm_pool_), sycl::backend::hip);
        }
#endif
        return interop_handle(static_cast<void*>(pool_), sycl::backend::host);
    }

    template <typename F>
    void paras_ext_interop_host_task(F& fn) {
        if (paras_recording()) paras_fail_recording("host_task(interop_handle)");
        wait_for_prior_commands();
        const native_device_scope scope(*this);
        fn(host_task_interop_handle());
    }

    template <typename F>
    bool paras_ext_capture_host_task(F&& fn) {
        if (paras_recording()) {
            auto held = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
            paras_record_host_task(std::function<void()>([held] { (*held)(); }));
            return true;
        }
        wait_for_prior_commands();
        return false;
    }

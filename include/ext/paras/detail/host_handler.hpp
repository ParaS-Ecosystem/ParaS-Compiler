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

inline void handler::wait_for_prior_commands() {
    wait_for_dependencies();
}

inline int handler::paras_activate_native_device() { return -1; }
inline void handler::paras_restore_native_device(int) noexcept {}
inline bool handler::paras_recording() const { return false; }
inline void handler::paras_record_host_task(std::function<void()> fn) { fn(); }
inline void handler::paras_fail_recording(const char* what) const {
    throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                          std::string(what) + " cannot be recorded into a graph");
}
inline void handler::paras_prefetch(const void*, std::size_t) {}
inline void handler::paras_mem_advise(const void*, std::size_t, int) {}

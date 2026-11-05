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

    void paras_detach_pool() {
#if (PARAS_CUDA_BACKEND)
        gpu_pool_.reset();
        paras_ext_slot_.detach();
#elif (PARAS_HIP_BACKEND)
        rocm_pool_.reset();
        paras_ext_slot_.detach();
#endif
    }

    ext::paras::queue_state ext_paras_get_state() const;
    event ext_paras_graph(const ext::paras::command_graph<ext::paras::graph_state::executable>& graph);
    void paras_begin_recording() const;
    std::shared_ptr<void> paras_end_recording() const;
    std::shared_ptr<void> paras_instantiate_graph(void* graph) const;

    event paras_submit_event() const;
    void paras_ext_share_pool() const;

private:
#if (PARAS_CUDA_BACKEND)
    cuda_threadpool& paras_ext_pool() const;
    mutable ext::paras::detail::pool_slot<cuda_threadpool> paras_ext_slot_{};
#elif (PARAS_HIP_BACKEND)
    rocm_threadpool& paras_ext_pool() const;
    mutable ext::paras::detail::pool_slot<rocm_threadpool> paras_ext_slot_{};
#endif

public:

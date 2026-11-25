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


#ifndef __SYCL_EXT_PARAS_COMMAND_GRAPH_IMPL_HPP__
#define __SYCL_EXT_PARAS_COMMAND_GRAPH_IMPL_HPP__

#include "kem_gpu/gpu_threadpool.hpp"
#include "ext/paras/command_graph.hpp"

#if (PARAS_CUDA_BACKEND)

inline void cuda_threadpool::paras_begin_recording() {
    if (props_.has_profiling())
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              "ParaS: a profiling queue cannot record a command_graph");
    if (recording_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "the queue is already recording a command_graph");
    ensure_stream();
    paras_cuda_detail::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
                             "cudaStreamBeginCapture");
    record_->error.clear();
    record_->host_tasks.clear();
    recording_ = true;
}

inline std::shared_ptr<void> cuda_threadpool::paras_end_recording() {
    if (!recording_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "the queue is not recording a command_graph");
    recording_ = false;
    ensure_stream();
    cudaGraph_t graph = nullptr;
    const cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess || graph == nullptr || !record_->error.empty()) {
        if (graph != nullptr) (void)cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "command_graph recording failed: " +
                                  (record_->error.empty() ? std::string(cudaGetErrorString(err))
                                                         : record_->error));
    }
    auto tasks = std::make_shared<std::vector<std::shared_ptr<void>>>(std::move(record_->host_tasks));
    record_->host_tasks.clear();
    return std::shared_ptr<void>(graph, [tasks](void* g) { (void)cudaGraphDestroy(static_cast<cudaGraph_t>(g)); });
}

inline std::shared_ptr<void> cuda_threadpool::paras_instantiate_graph(void* graph) {
    ensure_stream();
    cudaGraphExec_t exec = nullptr;
    paras_cuda_detail::check(
        cudaGraphInstantiateWithFlags(&exec, static_cast<cudaGraph_t>(graph), 0),
        "cudaGraphInstantiateWithFlags");
    const int device = dev_.get_native_id();
    return std::shared_ptr<void>(exec, [device](void* e) {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
        (void)cudaSetDevice(device);
        (void)cudaDeviceSynchronize();
        (void)cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(e));
        if (previous >= 0 && previous != device) (void)cudaSetDevice(previous);
    });
}

inline sycl::event cuda_threadpool::paras_launch_graph(void* exec) {
    if (recording_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              "ParaS: a command_graph cannot be submitted while recording");
    const std::uint64_t paras_t0 = paras_profiling_start();
    ensure_stream();
    paras_cuda_detail::check(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec), stream),
                             "cudaGraphLaunch");
    if (!async_launch())
        paras_cuda_detail::check(cudaStreamSynchronize(stream), "command_graph execution failed");
    return paras_finish_event(paras_t0);
}

inline sycl::ext::paras::queue_state cuda_threadpool::ext_paras_get_state() const {
    return recording_ ? sycl::ext::paras::queue_state::recording
                      : sycl::ext::paras::queue_state::executing;
}

inline sycl::event cuda_threadpool::ext_paras_graph(
    const sycl::ext::paras::command_graph<
        sycl::ext::paras::graph_state::executable>& graph) {
    if (!(graph.paras_device() == dev_))
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "the command_graph was recorded for another device");
    return paras_launch_graph(graph.paras_native());
}

#define PARAS_GRAPH_POOL get_or_create_gpu_pool()

#elif (PARAS_HIP_BACKEND)

inline void rocm_threadpool::paras_begin_recording() {
    if (props_.has_profiling())
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              "ParaS: a profiling queue cannot record a command_graph");
    if (recording_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "the queue is already recording a command_graph");
    ensure_stream();
    paras_rocm_detail::check(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed),
                             "hipStreamBeginCapture");
    record_->error.clear();
    record_->host_tasks.clear();
    recording_ = true;
}

inline std::shared_ptr<void> rocm_threadpool::paras_end_recording() {
    if (!recording_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "the queue is not recording a command_graph");
    recording_ = false;
    ensure_stream();
    hipGraph_t graph = nullptr;
    const hipError_t err = hipStreamEndCapture(stream, &graph);
    if (err != hipSuccess || graph == nullptr || !record_->error.empty()) {
        if (graph != nullptr) (void)hipGraphDestroy(graph);
        (void)hipGetLastError();
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "command_graph recording failed: " +
                                  (record_->error.empty() ? std::string(hipGetErrorString(err))
                                                         : record_->error));
    }
    auto tasks = std::make_shared<std::vector<std::shared_ptr<void>>>(std::move(record_->host_tasks));
    record_->host_tasks.clear();
    return std::shared_ptr<void>(graph, [tasks](void* g) { (void)hipGraphDestroy(static_cast<hipGraph_t>(g)); });
}

inline std::shared_ptr<void> rocm_threadpool::paras_instantiate_graph(void* graph) {
    ensure_stream();
    hipGraphExec_t exec = nullptr;
    paras_rocm_detail::check(
        hipGraphInstantiate(&exec, static_cast<hipGraph_t>(graph), nullptr, nullptr, 0),
        "hipGraphInstantiate");
    const int device = dev_.get_native_id();
    return std::shared_ptr<void>(exec, [device](void* e) {
        int previous = -1;
        if (hipGetDevice(&previous) != hipSuccess) previous = -1;
        (void)hipSetDevice(device);
        (void)hipDeviceSynchronize();
        (void)hipGraphExecDestroy(static_cast<hipGraphExec_t>(e));
        if (previous >= 0 && previous != device) (void)hipSetDevice(previous);
    });
}

inline sycl::event rocm_threadpool::paras_launch_graph(void* exec) {
    if (recording_)
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              "ParaS: a command_graph cannot be submitted while recording");
    const std::uint64_t paras_t0 = paras_profiling_start();
    ensure_stream();
    paras_rocm_detail::check(hipGraphLaunch(static_cast<hipGraphExec_t>(exec), stream),
                             "hipGraphLaunch");
    if (!async_launch())
        paras_rocm_detail::check(hipStreamSynchronize(stream), "command_graph execution failed");
    return paras_finish_event(paras_t0);
}

inline sycl::ext::paras::queue_state rocm_threadpool::ext_paras_get_state() const {
    return recording_ ? sycl::ext::paras::queue_state::recording
                      : sycl::ext::paras::queue_state::executing;
}

inline sycl::event rocm_threadpool::ext_paras_graph(
    const sycl::ext::paras::command_graph<
        sycl::ext::paras::graph_state::executable>& graph) {
    if (!(graph.paras_device() == dev_))
        throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                              "the command_graph was recorded for another device");
    return paras_launch_graph(graph.paras_native());
}

#define PARAS_GRAPH_POOL get_or_create_rocm_pool()

#endif

#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)

namespace paras_graph_detail {
[[noreturn]] inline void host_device() {
    throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                          "ParaS: command graphs need a CUDA or HIP device");
}
}

inline sycl::ext::paras::queue_state sycl::queue::ext_paras_get_state() const {
    if (!dev_.is_gpu()) return ext::paras::queue_state::executing;
    return PARAS_GRAPH_POOL.ext_paras_get_state();
}

inline void sycl::queue::paras_begin_recording() const {
    if (!dev_.is_gpu()) paras_graph_detail::host_device();
    PARAS_GRAPH_POOL.paras_begin_recording();
}

inline std::shared_ptr<void> sycl::queue::paras_end_recording() const {
    if (!dev_.is_gpu()) paras_graph_detail::host_device();
    return PARAS_GRAPH_POOL.paras_end_recording();
}

inline std::shared_ptr<void> sycl::queue::paras_instantiate_graph(void* graph) const {
    if (!dev_.is_gpu()) paras_graph_detail::host_device();
    return PARAS_GRAPH_POOL.paras_instantiate_graph(graph);
}

inline sycl::event sycl::queue::ext_paras_graph(
    const ext::paras::command_graph<
        ext::paras::graph_state::executable>& graph) {
    if (!dev_.is_gpu()) paras_graph_detail::host_device();
    return PARAS_GRAPH_POOL.ext_paras_graph(graph);
}

#undef PARAS_GRAPH_POOL

#endif

#endif

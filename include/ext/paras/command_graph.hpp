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


#ifndef __SYCL_EXT_PARAS_COMMAND_GRAPH_HPP__
#define __SYCL_EXT_PARAS_COMMAND_GRAPH_HPP__

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "sycl/context.hpp"
#include "sycl/device.hpp"
#include "sycl/event.hpp"
#include "sycl/exception.hpp"
#include "ext/paras/command_graph_types.hpp"
#include "sycl/property_list.hpp"
#include "sycl/queue.hpp"

#define SYCL_EXT_PARAS_COMMAND_GRAPH 1

namespace sycl {
namespace ext {
namespace paras {

template <graph_state State = graph_state::modifiable>
class command_graph;

template <>
class command_graph<graph_state::executable> {
public:
    command_graph() = delete;

    void update(const command_graph<graph_state::modifiable>&) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              "ParaS: command_graph::update is not supported");
    }

    void* paras_native() const noexcept { return exec_.get(); }
    const sycl::device& paras_device() const noexcept { return device_; }

private:
    friend class command_graph<graph_state::modifiable>;

    command_graph(std::shared_ptr<void> graph, std::shared_ptr<void> exec,
                  const sycl::device& device)
        : graph_(std::move(graph)), exec_(std::move(exec)), device_(device) {}

    std::shared_ptr<void> graph_;
    std::shared_ptr<void> exec_;
    sycl::device device_;
};

namespace paras_detail {
template <typename Queue, typename = void>
struct is_queue : std::false_type {};
template <typename Queue>
struct is_queue<Queue, std::void_t<decltype(std::declval<Queue&>().paras_begin_recording())>>
    : std::true_type {};
}

template <>
class command_graph<graph_state::modifiable> {
public:
    command_graph(const sycl::context&, const sycl::device& device,
                  const sycl::property_list& = {})
        : impl_(std::make_shared<impl>(device)) {}

    template <typename Queue, typename = std::enable_if_t<paras_detail::is_queue<Queue>::value>>
    explicit command_graph(const Queue& q, const sycl::property_list& = {})
        : impl_(std::make_shared<impl>(q.get_device())) {}

    template <typename Queue, typename = std::enable_if_t<paras_detail::is_queue<Queue>::value>>
    void begin_recording(Queue& q, const sycl::property_list& = {}) {
        if (impl_->recording)
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "command_graph is already recording");
        if (impl_->graph)
            throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                                  "ParaS: a command_graph is recorded once");
        if (!(q.get_device() == impl_->device))
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "the queue's device is not the graph's device");
        q.paras_begin_recording();
        auto queue = std::make_shared<Queue>(q);
        impl_->queue_identity = q.paras_identity();
        impl_->end = [queue] { return queue->paras_end_recording(); };
        impl_->instantiate = [queue](void* graph) { return queue->paras_instantiate_graph(graph); };
        impl_->recording = true;
    }

    template <typename Queue>
    void begin_recording(const std::vector<Queue>& queues, const sycl::property_list& props = {}) {
        if (queues.size() != 1)
            throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                                  "ParaS: a command_graph records one queue");
        begin_recording(const_cast<Queue&>(queues.front()), props);
    }

    void end_recording() {
        if (!impl_->recording) return;
        impl_->recording = false;
        impl_->graph = impl_->end();
    }

    template <typename Queue, typename = std::enable_if_t<paras_detail::is_queue<Queue>::value>>
    void end_recording(Queue& q) {
        if (impl_->recording && q.paras_identity() != impl_->queue_identity)
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "the queue is not recording to this graph");
        end_recording();
    }

    template <typename Queue>
    void end_recording(const std::vector<Queue>& queues) {
        for (const Queue& q : queues) end_recording(const_cast<Queue&>(q));
    }

    command_graph<graph_state::executable> finalize(const sycl::property_list& = {}) const {
        if (impl_->recording)
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "command_graph::finalize while recording");
        if (!impl_->graph)
            throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                  "command_graph::finalize before anything was recorded");
        return command_graph<graph_state::executable>(
            impl_->graph, impl_->instantiate(impl_->graph.get()), impl_->device);
    }

private:
    struct impl {
        explicit impl(const sycl::device& d) : device(d) {}
        impl(const impl&) = delete;
        impl& operator=(const impl&) = delete;
        ~impl() {
            if (!recording || !end) return;
            try {
                (void)end();
            } catch (...) {
            }
        }
        sycl::device device;
        const void* queue_identity = nullptr;
        std::function<std::shared_ptr<void>()> end;
        std::function<std::shared_ptr<void>(void*)> instantiate;
        std::shared_ptr<void> graph;
        bool recording = false;
    };
    std::shared_ptr<impl> impl_;
};

command_graph(const sycl::context&, const sycl::device&, const sycl::property_list& = {})
    -> command_graph<graph_state::modifiable>;
template <typename Queue>
command_graph(const Queue&, const sycl::property_list& = {})
    -> command_graph<graph_state::modifiable>;

}
}
}

#if !PARAS_GPU_BACKEND
inline sycl::ext::paras::queue_state sycl::queue::ext_paras_get_state() const {
    return ext::paras::queue_state::executing;
}
inline void sycl::queue::paras_begin_recording() const {
    throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                          "ParaS: command graphs need a CUDA or HIP device");
}
inline std::shared_ptr<void> sycl::queue::paras_end_recording() const { return {}; }
inline std::shared_ptr<void> sycl::queue::paras_instantiate_graph(void*) const { return {}; }
inline sycl::event sycl::queue::ext_paras_graph(
    const ext::paras::command_graph<
        ext::paras::graph_state::executable>&) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                          "ParaS: command graphs need a CUDA or HIP device");
}
#endif

#endif

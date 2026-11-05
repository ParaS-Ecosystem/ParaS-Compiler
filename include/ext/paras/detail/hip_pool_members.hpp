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

public:
    void paras_ext_adopt(const rocm_threadpool& other) {
        stream = other.stream;
        ownsStream = other.ownsStream;
        stream_alive_ = other.stream_alive_;
        stream_owner_ = other.stream_owner_;
        recording_ = other.recording_;
        record_ = other.record_;
    }
    void paras_ext_take(rocm_threadpool& other) {
        stream_alive_ = std::move(other.stream_alive_);
        stream_owner_ = std::move(other.stream_owner_);
        recording_ = other.recording_;
        record_ = other.record_;
    }
    sycl::event paras_stream_event() const {
        hipStream_t s = stream;
        std::shared_ptr<bool> alive = stream_alive_;
        return sycl::event::paras_async(
            sycl::backend::hip,
            [s, alive] {
                if (!(alive && *alive)) return;
                if (stream_recording(s))
                    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                          "event::wait on a command recorded into a graph");
                const hipError_t err = hipStreamSynchronize(s);
                if (err != hipSuccess)
                    throw sycl::exception(std::string("kernel execution failed: ") +
                                          hipGetErrorString(err));
            },
            [s, alive] {
                return !(alive && *alive) ||
                       (!stream_recording(s) && hipStreamQuery(s) != hipErrorNotReady);
            });
    }
    bool paras_profiling() const noexcept { return props_.has_profiling(); }
    bool async_launch() const { return (async_launch_ || recording_) && !props_.has_profiling(); }
    bool is_recording() const noexcept { return recording_; }

    bool paras_ext_release_owner() noexcept {
        if (!stream_owner_) return false;
        stream_owner_.reset();
        stream = nullptr;
        ownsStream = false;
        return true;
    }

    void paras_order_streams() {
        if (!async_launch_ || recording_ || stream == nullptr) return;
        struct last_stream {
            std::mutex mutex;
            hipStream_t stream = nullptr;
            std::shared_ptr<bool> alive;
        };
        static last_stream last;
        std::lock_guard<std::mutex> lock(last.mutex);
        if (last.stream == stream) return;
        if (last.alive && *last.alive && !stream_recording(last.stream)) {
            const hipError_t err = hipStreamSynchronize(last.stream);
            if (err != hipSuccess) (void)hipGetLastError();
        }
        last.stream = stream;
        last.alive = stream_alive_;
    }

    void paras_record_host_task(std::function<void()> fn) {
        auto task = std::make_shared<std::function<void()>>(std::move(fn));
        const hipError_t err = hipLaunchHostFunc(
            stream, [](void* p) { (*static_cast<std::function<void()>*>(p))(); }, task.get());
        if (err != hipSuccess) fail_recording("a host task (hipLaunchHostFunc failed)");
        record_->host_tasks.push_back(std::move(task));
    }
    [[noreturn]] void paras_fail_recording(const char* what) const { fail_recording(what); }
    void paras_begin_recording();
    std::shared_ptr<void> paras_end_recording();
    std::shared_ptr<void> paras_instantiate_graph(void* graph);
    sycl::event paras_launch_graph(void* exec);
    sycl::ext::paras::queue_state ext_paras_get_state() const;
    sycl::event ext_paras_graph(const sycl::ext::paras::command_graph<
                                sycl::ext::paras::graph_state::executable>& graph);

    const hipDeviceProp_t& device_props() {
        if (!device_props_valid_) {
            const hipError_t err = hipGetDeviceProperties(&device_props_, dev_.get_native_id());
            if (err != hipSuccess)
                throw sycl::exception(std::string("hipGetDeviceProperties failed: ") +
                                      hipGetErrorString(err));
            device_props_valid_ = true;
        }
        return device_props_;
    }

    sycl::event prefetch(const void* ptr, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (numBytes != 0 && !recording_ &&
            hipMemPrefetchAsync(ptr, numBytes, dev_.get_native_id(), stream) != hipSuccess)
            (void)hipGetLastError();
        if (!async_launch()) (void)hipStreamSynchronize(stream);
        return paras_finish_event(paras_t0);
    }

    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (numBytes != 0 && !recording_ &&
            hipMemAdvise(ptr, numBytes, static_cast<hipMemoryAdvise>(advice), dev_.get_native_id()) != hipSuccess)
            (void)hipGetLastError();
        return paras_finish_event(paras_t0);
    }

    sycl::event prefetch(const void* ptr, size_t numBytes, sycl::event dep) {
        dep.wait();
        return prefetch(ptr, numBytes);
    }
    sycl::event prefetch(const void* ptr, size_t numBytes, const std::vector<sycl::event>& deps) {
        sycl::event::wait(deps);
        return prefetch(ptr, numBytes);
    }
    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice, sycl::event dep) {
        dep.wait();
        return mem_advise(ptr, numBytes, advice);
    }
    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice,
                           const std::vector<sycl::event>& deps) {
        sycl::event::wait(deps);
        return mem_advise(ptr, numBytes, advice);
    }

    bool stream_ordered_copy(void* dest, const void* src) const {
        if (!async_launch()) return false;
        return is_device_pointer(dest) && is_device_pointer(src);
    }

    static bool is_device_pointer(const void* p) {
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, p) != hipSuccess) {
            (void)hipGetLastError();
            return false;
        }
        return attr.type == hipMemoryTypeDevice;
    }

    template <typename Func>
    void paras_ext_nd_range_1D(const sycl::nd_range<1>& r, Func f, std::size_t sharedMemoryBytes);
    template <typename Func>
    void paras_ext_nd_range_2D(const sycl::nd_range<2>& r, Func f, std::size_t sharedMemoryBytes);
    template <typename Func>
    void paras_ext_nd_range_3D(const sycl::nd_range<3>& r, Func f, std::size_t sharedMemoryBytes);

private:
    std::shared_ptr<bool> stream_alive_;
    std::shared_ptr<void> stream_owner_;
    hipDeviceProp_t device_props_{};
    bool device_props_valid_ = false;
    bool async_launch_ = [] {
        const char* env = std::getenv("PARAS_ASYNC_KERNELS");
        return env != nullptr && std::string(env) == "1";
    }();
    sycl::ext::paras::detail::shared_flag recording_;
    std::shared_ptr<sycl::ext::paras::detail::record_state> record_ =
        std::make_shared<sycl::ext::paras::detail::record_state>();

    [[noreturn]] void fail_recording(const char* what) const {
        record_->error = std::string(what) + " cannot be recorded into a graph";
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              record_->error);
    }

    static bool stream_recording(hipStream_t s) {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        return hipStreamIsCapturing(s, &status) == hipSuccess &&
               status == hipStreamCaptureStatusActive;
    }

    void paras_ext_own_stream() {
        stream_alive_ = std::make_shared<bool>(true);
        const int device = dev_.is_gpu() ? dev_.get_native_id() : -1;
        std::shared_ptr<bool> alive = stream_alive_;
        stream_owner_ = std::shared_ptr<void>(static_cast<void*>(stream), [device, alive](void* p) {
            hipStream_t s = static_cast<hipStream_t>(p);
            int previous = -1;
            if (device >= 0) {
                if (hipGetDevice(&previous) != hipSuccess) previous = -1;
                const hipError_t setErr = hipSetDevice(device);
                if (setErr != hipSuccess)
                    std::cerr << "hipSetDevice before hipStreamDestroy failed: "
                              << hipGetErrorString(setErr) << '\n';
            }
            const hipError_t syncErr = hipStreamSynchronize(s);
            if (syncErr != hipSuccess)
                std::cerr << "hipStreamSynchronize before destroy failed: "
                          << hipGetErrorString(syncErr) << '\n';
            *alive = false;
            const hipError_t err = hipStreamDestroy(s);
            if (err != hipSuccess)
                std::cerr << "hipStreamDestroy failed: " << hipGetErrorString(err) << '\n';
            if (previous >= 0 && previous != device) (void)hipSetDevice(previous);
        });
    }

public:

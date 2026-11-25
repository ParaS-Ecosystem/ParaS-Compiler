# ParaS extensions

These extensions follow the SYCL 2020 extension rules: names are in `sycl::ext::paras`, member functions start with `ext_paras_`, and feature macros start with `SYCL_EXT_PARAS_`. Without them, ParaS behaves as it did before: nothing changes unless a program asks for it.

## Command graphs: `SYCL_EXT_PARAS_COMMAND_GRAPH`

Record commands once and replay them cheaply. Header: `ext/paras/command_graph.hpp`, also included by `sycl/sycl.hpp`.

```cpp
namespace ext = sycl::ext::paras;
ext::command_graph graph(q.get_context(), q.get_device());
graph.begin_recording(q);
q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = in[i] * 2; });
graph.end_recording(q);
auto exec = graph.finalize();
q.ext_paras_graph(exec);
q.wait();
```

- CUDA and HIP devices only. Other devices throw `errc::feature_not_supported`.
- A graph records one queue, and is recorded once. `command_graph::update` isn't supported.
- A profiling queue can't record.
- While recording, these throw: `queue::wait`, copies to or from host memory, and `host_task(interop_handle)`.
- `queue::ext_paras_get_state()` returns `queue_state::recording` or `queue_state::executing`.

## Asynchronous launches: `PARAS_ASYNC_KERNELS=1`

Set this environment variable to return from kernel launches, memsets and device-to-device copies without waiting. Events then wait on the queue's stream. It has no effect on a profiling queue, and it's off by default.

When a queue launches after another queue, ParaS first waits for the other queue's stream. So commands on different queues stay ordered, but those queues don't overlap in this mode.

## Occupancy hint

A kernel functor used with a 1-D `nd_range` on HIP can declare the following, and ParaS applies them as launch bounds:

```cpp
static constexpr int paras_max_work_group_size = 256;
static constexpr int paras_min_waves_per_eu = 4;
```

A local range above `paras_max_work_group_size` throws `errc::nd_range`.

## Core SYCL 2020 behaviour added

- `handler::host_task` accepts a functor taking `interop_handle`, and runs after commands already submitted to the queue. The queue's device is the current native device while it runs, and also during `parasSYCL_enqueue_custom_operation`.
- `handler::prefetch` and `handler::mem_advise` reach the GPU runtime. `queue::prefetch` and `queue::mem_advise` remain no-op hints, which SYCL 2020 permits; use the handler forms to reach the runtime.
- `info::device::sub_group_sizes` reports 32 for AMD GPUs whose architecture has a four-digit `gfx` name.
- Copies of a queue share one stream.

## Device visibility

`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are honoured on AMD, and `CUDA_VISIBLE_DEVICES` accepts `GPU-` UUIDs on NVIDIA. `parascc` itself needs `-fno-rtti` in `CMAKE_CXX_FLAGS` when LLVM was built without RTTI.

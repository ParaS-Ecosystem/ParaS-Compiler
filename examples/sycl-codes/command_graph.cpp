#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

namespace sycl_exp = sycl::ext::paras;

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    constexpr std::size_t n = 1 << 16;
    int* in = sycl::malloc_device<int>(n, q);
    int* out = sycl::malloc_device<int>(n, q);
    int* tmp = sycl::malloc_device<int>(n, q);
    int failures = 0;

    sycl_exp::command_graph graph{q.get_context(), q.get_device()};
    graph.begin_recording(q);
    if (q.ext_paras_get_state() != sycl_exp::queue_state::recording) {
        std::printf("FAIL: queue state is not recording\n");
        ++failures;
    }
    q.memset(tmp, 0, n * sizeof(int));
    q.parasSYCL_enqueue_custom_operation([&](sycl::interop_handle ih) {
#if PARAS_HIP_BACKEND
        (void)hipMemsetAsync(out, 0, n * sizeof(int), ih.get_native_queue<sycl::backend::hip>());
#elif PARAS_CUDA_BACKEND
        (void)cudaMemsetAsync(out, 0, n * sizeof(int), ih.get_native_queue<sycl::backend::cuda>());
#endif
    });
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { tmp[i[0]] += in[i[0]] + 1; });
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(256)),
                   [=](sycl::nd_item<1> it) {
                       const std::size_t i = it.get_global_id(0);
                       tmp[i] = 2 * tmp[i] + 3;
                   });
    q.memcpy(out, tmp, n * sizeof(int));
    graph.end_recording(q);
    if (q.ext_paras_get_state() != sycl_exp::queue_state::executing) {
        std::printf("FAIL: queue state is not executing after end_recording\n");
        ++failures;
    }
    auto exec = graph.finalize();

    std::vector<int> host(n);
    for (int round = 0; round < 3; ++round) {
        std::vector<int> input(n);
        for (std::size_t i = 0; i < n; ++i) input[i] = static_cast<int>(i) * (round + 1);
        q.memcpy(in, input.data(), n * sizeof(int));
        q.ext_paras_graph(exec);
        q.memcpy(host.data(), out, n * sizeof(int));
        q.wait();
        for (std::size_t i = 0; i < n && failures < 5; ++i) {
            const int want = 2 * (input[i] + 1) + 3;
            if (host[i] != want) {
                std::printf("FAIL: round %d out[%zu] = %d, want %d\n", round, i, host[i], want);
                ++failures;
            }
        }
    }

    auto refused = [&](const char* what, auto command) {
        sycl_exp::command_graph g{q.get_context(), q.get_device()};
        g.begin_recording(q);
        bool threw = false;
        try {
            command();
        } catch (const sycl::exception&) {
            threw = true;
        }
        bool end_threw = false;
        try {
            g.end_recording(q);
        } catch (const sycl::exception&) {
            end_threw = true;
        }
        if (!threw || !end_threw) {
            std::printf("FAIL: %s while recording: threw %d, end_recording threw %d\n", what,
                        threw, end_threw);
            ++failures;
        }
    };
    refused("queue::wait", [&] { q.wait(); });
    refused("copy to host", [&] { q.memcpy(host.data(), out, sizeof(int)); });
    refused("host_task(interop_handle)",
            [&] { q.submit([&](sycl::handler& h) { h.host_task([](sycl::interop_handle) {}); }); });
    q.memcpy(host.data(), out, sizeof(int));
    q.wait();

    {
        int* counter = sycl::malloc_shared<int>(1, q);
        *counter = 0;
        std::vector<int> seen;
        sycl_exp::command_graph g{q.get_context(), q.get_device()};
        g.begin_recording(q);
        q.single_task([=] { *counter += 1; });
        q.submit([&](sycl::handler& h) { h.host_task([&seen, counter] { seen.push_back(*counter); }); });
        q.single_task([=] { *counter += 10; });
        g.end_recording(q);
        auto host_exec = g.finalize();
        for (int r = 0; r < 3; ++r) q.ext_paras_graph(host_exec);
        q.wait();
        const std::vector<int> want = {1, 12, 23};
        if (seen != want || *counter != 33) {
            std::printf("FAIL: host node saw %zu values (", seen.size());
            for (int v : seen) std::printf(" %d", v);
            std::printf(" ), counter %d; want 1 12 23, 33\n", *counter);
            ++failures;
        }
        sycl::free(counter, q);
    }

    {
        int* counter = sycl::malloc_shared<int>(1, q);
        *counter = 0;
        auto ran = std::make_shared<int>(0);
        {
            sycl_exp::command_graph g{q.get_context(), q.get_device()};
            g.begin_recording(q);
            q.single_task([=] { *counter += 1; });
            q.submit([&](sycl::handler& h) {
                h.host_task([owned = std::make_unique<int>(7), ran] { *ran += *owned; });
            });
            g.end_recording(q);
            auto exec = g.finalize();
            q.ext_paras_graph(exec);
        }
        q.wait();
        if (*ran != 7 || *counter != 1) {
            std::printf("FAIL: move-only host node ran %d (want 7), counter %d (want 1)\n", *ran, *counter);
            ++failures;
        }
        sycl::free(counter, q);
    }

    constexpr int kernels = 200, reps = 20;
    auto chain = [&] {
        for (int k = 0; k < kernels; ++k)
            q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) { tmp[i[0]] += 1; });
    };
    sycl_exp::command_graph chain_graph{q};
    chain_graph.begin_recording(q);
    chain();
    chain_graph.end_recording();
    auto chain_exec = chain_graph.finalize();
    chain();
    q.ext_paras_graph(chain_exec);
    q.wait();
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    for (int r = 0; r < reps; ++r) chain();
    q.wait();
    auto t1 = clock::now();
    for (int r = 0; r < reps; ++r) q.ext_paras_graph(chain_exec);
    q.wait();
    auto t2 = clock::now();
    const double eager_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    const double graph_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / reps;
    std::printf("%d kernels: one by one %.1f us, one replay %.1f us\n", kernels, eager_us,
                graph_us);

    sycl::free(in, q);
    sycl::free(out, q);
    sycl::free(tmp, q);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

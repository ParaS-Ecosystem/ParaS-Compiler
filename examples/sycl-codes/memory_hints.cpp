#include <sycl/sycl.hpp>

#include <cstdio>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    constexpr std::size_t n = 1 << 20;
    constexpr int kReadMostly = 1;
    int failures = 0;

    float* shared = sycl::malloc_shared<float>(n, q);
    for (std::size_t i = 0; i < n; ++i) shared[i] = static_cast<float>(i);
    float* dev = sycl::malloc_device<float>(n, q);

    q.mem_advise(shared, n * sizeof(float), kReadMostly);
    q.prefetch(shared, n * sizeof(float));
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { dev[i] = 2.f * shared[i] + 1.f; });
    sycl::event e = q.prefetch(dev, n * sizeof(float));
    q.mem_advise(dev, n * sizeof(float), kReadMostly, e);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { shared[i] = dev[i] - 1.f; });
    q.wait();
    for (std::size_t i = 0; i < n && failures < 5; ++i) {
        const float want = 2.f * static_cast<float>(i);
        if (shared[i] != want) {
            std::printf("FAIL: shared[%zu] = %g, want %g\n", i, shared[i], want);
            ++failures;
        }
    }

    q.submit([&](sycl::handler& h) { h.prefetch(shared, n * sizeof(float)); });
    q.submit([&](sycl::handler& h) { h.mem_advise(shared, n * sizeof(float), kReadMostly); });
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { shared[i] += 1.f; });
    q.wait();
    if (shared[7] != 15.f) {
        std::printf("FAIL: after the handler hints shared[7] = %g, want 15\n", shared[7]);
        ++failures;
    }

    sycl::free(shared, q);
    sycl::free(dev, q);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

#include <sycl/sycl.hpp>

#include <cstdio>
#include <vector>

namespace {
constexpr std::size_t n = 1 << 24;

int check(const std::vector<int>& h, int (*want)(std::size_t), const char* what) {
    for (std::size_t i = 0; i < h.size(); ++i)
        if (h[i] != want(i)) {
            std::printf("FAIL %s: [%zu] = %d, want %d\n", what, i, h[i], want(i));
            return 1;
        }
    return 0;
}
}

int main() {
    int failures = 0;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    sycl::queue q2 = q;
    int* x = sycl::malloc_device<int>(n, q);
    int* y = sycl::malloc_device<int>(n, q);
    std::vector<int> h(n);

    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        int v = 0;
        for (int r = 0; r < 256; ++r) v += static_cast<int>((i[0] + r) % 7);
        x[i[0]] = v;
    });
    q2.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { y[i[0]] = x[i[0]] + 1; });
    q.memcpy(h.data(), y, n * sizeof(int));
    q.wait();
    failures += check(h, [](std::size_t i) {
        int v = 0;
        for (int r = 0; r < 256; ++r) v += static_cast<int>((i + r) % 7);
        return v + 1;
    }, "copy ordering");

    sycl::event e = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i[0]] = static_cast<int>(i[0] % 11); });
    });
    e.wait();
    if (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
        std::printf("FAIL event: not complete after wait()\n");
        ++failures;
    }

    sycl::queue q3{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    sycl::event produced = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        int v = 0;
        for (int r = 0; r < 256; ++r) v += static_cast<int>((i[0] * 3 + r) % 5);
        x[i[0]] = v;
    });
    q3.parallel_for(sycl::range<1>(n), produced, [=](sycl::id<1> i) { y[i[0]] = 2 * x[i[0]]; });
    q3.wait();
    q.memcpy(h.data(), y, n * sizeof(int));
    q.wait();
    failures += check(h, [](std::size_t i) {
        int v = 0;
        for (int r = 0; r < 256; ++r) v += static_cast<int>((i * 3 + r) % 5);
        return 2 * v;
    }, "dependency event");

    sycl::free(x, q);
    sycl::free(y, q);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

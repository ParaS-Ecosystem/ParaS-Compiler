#include <sycl/sycl.hpp>

#include <cstdio>
#include <vector>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    constexpr std::size_t n = 1 << 20;
    int* a = sycl::malloc_device<int>(n, q);
    int* b = sycl::malloc_device<int>(n, q);
    int failures = 0;

    q.memset(a, 0, n * sizeof(int));
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a[i[0]] += static_cast<int>(i[0]); });
    q.parallel_for(sycl::range<2>(1024, n / 1024), [=](sycl::id<2> i) {
        a[i[0] * (n / 1024) + i[1]] += 1;
    });
    q.parallel_for(sycl::range<3>(16, 64, n / 1024), [=](sycl::id<3> i) {
        a[(i[0] * 64 + i[1]) * (n / 1024) + i[2]] += 1;
    });
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(256)),
                   [=](sycl::nd_item<1> it) { a[it.get_global_id(0)] += 1; });
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(1024, n / 1024), sycl::range<2>(16, 16)),
                   [=](sycl::nd_item<2> it) {
                       a[it.get_global_id(0) * (n / 1024) + it.get_global_id(1)] += 1;
                   });
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(16, 64, n / 1024), sycl::range<3>(1, 4, 64)),
                   [=](sycl::nd_item<3> it) {
                       a[(it.get_global_id(0) * 64 + it.get_global_id(1)) * (n / 1024) +
                         it.get_global_id(2)] += 1;
                   });
    q.memcpy(b, a, n * sizeof(int));
    sycl::event e = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i[0]] *= 2; });
    });
    e.wait();
    if (e.get_info<sycl::info::event::command_execution_status>() !=
        sycl::info::event_command_status::complete) {
        std::printf("FAIL: event not complete after wait()\n");
        ++failures;
    }

    std::vector<int> seen(1);
    q.submit([&](sycl::handler& h) { h.host_task([&] { seen[0] = 1; }); });
    q.submit([&](sycl::handler& h) {
        h.host_task([&](sycl::interop_handle ih) { (void)ih; seen[0] += 1; });
    });

    std::vector<int> host(n);
    q.memcpy(host.data(), b, n * sizeof(int));
    q.wait();
    for (std::size_t i = 0; i < n && failures < 5; ++i) {
        const int want = 2 * (static_cast<int>(i) + 5);
        if (host[i] != want) {
            std::printf("FAIL: b[%zu] = %d, want %d\n", i, host[i], want);
            ++failures;
        }
    }
    if (seen[0] != 2) {
        std::printf("FAIL: host tasks ran %d times\n", seen[0]);
        ++failures;
    }
    sycl::free(a, q);
    sycl::free(b, q);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

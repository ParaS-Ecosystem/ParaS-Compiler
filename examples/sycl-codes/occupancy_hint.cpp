#include <sycl/sycl.hpp>

#include <cstdio>
#include <vector>

namespace {
constexpr int kGroup = 256;
constexpr std::size_t kGroups = 4096;

template <bool Hint>
struct GroupSum {
    const int* in;
    int* out;
    sycl::local_accessor<int, 1> lds;

    PARAS_KERNEL_HD void operator()(sycl::nd_item<1> it) const {
        const int l = static_cast<int>(it.get_local_id(0));
        const std::size_t g = it.get_group(0);
        lds[l] = in[g * kGroup + l] * (l + 1);
        for (int s = kGroup / 2; s > 0; s /= 2) {
            sycl::group_barrier(it.get_group());
            if (l < s) lds[l] += lds[l + s];
        }
        if (l == 0) out[g] = lds[0];
    }
};

template <>
struct GroupSum<true> : GroupSum<false> {
    static constexpr int paras_max_work_group_size = kGroup;
    static constexpr int paras_min_waves_per_eu = 2;
};

template <bool Hint>
std::vector<int> run(sycl::queue& q, const int* in, int* out, std::size_t local) {
    q.submit([&](sycl::handler& cgh) {
        const GroupSum<false> base{in, out, sycl::local_accessor<int, 1>(sycl::range<1>(kGroup), cgh)};
        const GroupSum<Hint> k{base};
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kGroups * local), sycl::range<1>(local)), k);
    });
    std::vector<int> h(kGroups);
    q.memcpy(h.data(), out, kGroups * sizeof(int));
    q.wait();
    return h;
}
}

int main() {
    int failures = 0;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    int* in = sycl::malloc_device<int>(kGroups * kGroup, q);
    int* out = sycl::malloc_device<int>(kGroups, q);
    q.parallel_for(sycl::range<1>(kGroups * kGroup), [=](sycl::id<1> i) {
        in[i[0]] = static_cast<int>(i[0] % 13) - 6;
    });
    q.wait();

    const std::vector<int> plain = run<false>(q, in, out, kGroup);
    const std::vector<int> hinted = run<true>(q, in, out, kGroup);
    for (std::size_t g = 0; g < kGroups; ++g) {
        int want = 0;
        for (int l = 0; l < kGroup; ++l) want += (static_cast<int>((g * kGroup + l) % 13) - 6) * (l + 1);
        if (plain[g] != want || hinted[g] != want) {
            std::printf("FAIL group %zu: %d / %d, want %d\n", g, plain[g], hinted[g], want);
            ++failures;
            break;
        }
    }

#if (PARAS_HIP_BACKEND)
    bool refused = false;
    try {
        run<true>(q, in, out, 2 * kGroup);
    } catch (const sycl::exception&) {
        refused = true;
    }
    if (!refused) {
        std::printf("FAIL a work-group larger than paras_max_work_group_size was launched\n");
        ++failures;
    }
#endif

    sycl::free(in, q);
    sycl::free(out, q);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

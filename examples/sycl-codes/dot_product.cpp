#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t N = 1 << 20;

  queue q;

  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  int *A = malloc_shared<int>(N, q);
  int *B = malloc_shared<int>(N, q);
  int *dot = malloc_shared<int>(1, q);

  for (size_t i = 0; i < N; i++) {
    A[i] = 1;
    B[i] = 2;
  }
  *dot = 0;

  q.submit([&](handler &h) {
    h.parallel_for<class kernel>(range<1>(N), [=](id<1> i) {
      atomic_ref<int, memory_order::relaxed, memory_scope::device,
                 access::address_space::global_space>
          atomic_dot(dot[0]);

      atomic_dot.fetch_add(A[i] * B[i]);
    });
  });

  q.wait();

  std::cout << "Dot product = " << *dot << "\n";
  std::cout << "Expected    = " << 2 * N << "\n";

  free(A, q);
  free(B, q);
  free(dot, q);

  return 0;
}

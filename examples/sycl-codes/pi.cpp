// ============================================================================
//                  π APPROXIMATION USING SYCL PARALLEL INTEGRATION (USM)
//
//   Method:
//      π = 4 * ∫₀¹ sqrt(1 - x²) dx
//
//   Using Riemann Sum with N intervals computed in parallel on SYCL device.
// ============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {

  // Number of intervals (increase for higher accuracy)
  const size_t n = 100000000; // 100 million intervals
  const double dx = 1.0 / n;

  // Select device
  queue q;
  cout << "SYCL Device: " << q.get_device().get_info<info::device::name>()
       << "\n\n";

  // Allocate USM shared memory
  double *results = malloc_shared<double>(n, q);

  // Initialize (optional)
  for (size_t i = 0; i < n; i++)
    results[i] = 0.0;

  // Submit kernel
  q.submit([&](handler &h) {
    h.parallel_for<class pi_kernel_usm>(range<1>(n), [=](id<1> i) {
      size_t idx = i[0];
      double x = (idx + 0.5) * dx;
      results[idx] = sycl::sqrt(1.0 - x * x) * dx;
    });
  });

  // Wait for kernel to finish
  q.wait();

  // Compute π on host
  double pi = 0.0;
  for (size_t i = 0; i < n; i++)
    pi += results[i];

  pi *= 4.0;

  // Free USM memory
  free(results, q);

  // Print result
  cout << fixed << setprecision(10);
  cout << "Intervals used      :  " << n << "\n";
  cout << "Approximation of π  :  " << pi << "\n";

  return 0;
}

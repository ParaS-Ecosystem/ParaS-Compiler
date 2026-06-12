#include <chrono>
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

#define PI sycl::acos(-1.0)
#define LINE "--------------------"

// FUNCTION DECLARATIONS
void initial_value(queue &q, unsigned int n, double dx, double length,
                   double *u);
void zero(queue &q, unsigned int n, double *u);
void solve(queue &q, unsigned int n, double alpha, double dx, double dt,
           double *u, double *u_tmp);
double solution(double t, double x, double y, double alpha, double length);
double l2norm(unsigned int n, const double *u, int nsteps, double dt,
              double alpha, double dx, double length);

int main(int argc, char *argv[]) {

  auto start = std::chrono::high_resolution_clock::now();

  unsigned int n = 6000;
  int nsteps = 100;

  if (argc == 3) {
    n = atoi(argv[1]);
    nsteps = atoi(argv[2]);
  }

  double alpha = 0.1;
  double length = 1000.0;
  double dx = length / (n + 1);
  double dt = 0.5 / nsteps;

  double r = alpha * dt / (dx * dx);

  queue q{};
  std::cout << "\n MMS heat equation\n\n"
            << LINE << "\n"
            << "Problem input\n\n"
            << " Grid size: " << n << " x " << n << "\n"
            << " Cell width: " << dx << "\n"
            << " Grid length: " << length << " x " << length << "\n\n"
            << " Alpha: " << alpha << "\n\n"
            << " Steps: " << nsteps << "\n"
            << " Total time: " << dt * (double)nsteps << "\n"
            << " Time step: " << dt << "\n"
            << " SYCL device: " << q.get_device().get_info<info::device::name>()
            << "\n"
            << LINE << "\n";

  std::cout << "Stability\n\n";
  std::cout << " r value: " << r << "\n";
  std::cout << LINE << "\n";

  // USM Allocations
  double *u = malloc_shared<double>(n * n, q);
  double *u_tmp = malloc_shared<double>(n * n, q);

  initial_value(q, n, dx, length, u);
  zero(q, n, u_tmp);
  q.wait();

  auto tic = std::chrono::high_resolution_clock::now();

  for (int t = 0; t < nsteps; ++t) {
    solve(q, n, alpha, dx, dt, u, u_tmp);

    double *temp = u;
    u = u_tmp;
    u_tmp = temp;
  }

  q.wait();
  auto toc = std::chrono::high_resolution_clock::now();

  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  auto stop = std::chrono::high_resolution_clock::now();

  std::cout
      << "Results\n\n"
      << "(L2norm): " << norm << "\n"
      << "Solve time (s): "
      << std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic)
             .count()
      << "\n"
      << "Total time (s): "
      << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
             .count()
      << "\n"
      << "Bandwidth (GB/s): "
      << 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) /
             std::chrono::duration_cast<std::chrono::duration<double>>(toc -
                                                                       tic)
                 .count()
      << "\n"
      << LINE << "\n";

  free(u, q);
  free(u_tmp, q);
  return 0;
}

// ---------------------- KERNELS ----------------------

void initial_value(queue &q, unsigned int n, double dx, double length,
                   double *u) {

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class InitialValueKernel>(range<2>(n, n), [=](id<2> idx) {
      size_t j = idx[0];
      size_t i = idx[1];

      double y = dx * (j + 1);
      double x = dx * (i + 1);

      u[j * n + i] = sycl::sin(PI * x / length) * sycl::sin(PI * y / length);
    });
  });
}

void zero(queue &q, unsigned int n, double *u) {

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class ZeroKernel>(
        range<2>(n, n), [=](id<2> idx) { u[idx[0] * n + idx[1]] = 0.0; });
  });
}

void solve(queue &q, unsigned int n, double alpha, double dx, double dt,
           double *u, double *u_tmp) {

  double r = alpha * dt / (dx * dx);
  double r2 = 1.0 - 4.0 * r;

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class SolveKernel>(range<2>(n, n), [=](id<2> idx) {
      size_t j = idx[0], i = idx[1];

      double center = u[j * n + i];

      double up = (j < n - 1) ? u[(j + 1) * n + i] : 0.0;
      double down = (j > 0) ? u[(j - 1) * n + i] : 0.0;
      double right = (i < n - 1) ? u[j * n + (i + 1)] : 0.0;
      double left = (i > 0) ? u[j * n + (i - 1)] : 0.0;

      u_tmp[j * n + i] = r2 * center + r * (up + down + left + right);
    });
  });
}

double solution(double t, double x, double y, double alpha, double length) {
  return sycl::exp(-2.0 * alpha * PI * PI * t / (length * length)) *
         sycl::sin(PI * x / length) * sycl::sin(PI * y / length);
}

double l2norm(unsigned int n, const double *u, int nsteps, double dt,
              double alpha, double dx, double length) {

  double time = dt * nsteps;

  double sum = 0.0;
  double y = dx;

  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double exact = solution(time, x, y, alpha, length);
      double diff = u[j * n + i] - exact;
      sum += diff * diff;
      x += dx;
    }
    y += dx;
  }

  return sycl::sqrt(sum);
}

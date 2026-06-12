// ============================================================================
//                          SYCL MATRIX MULTIPLICATION (USM)
// ============================================================================

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {

  size_t N;
  cout << "Enter matrix size N (NxN): ";
  cin >> N;

  srand(time(nullptr));

  // Create queue
  queue q{cpu_selector_v};
  cout << "\nSYCL Device: " << q.get_device().get_info<info::device::name>()
       << "\n\n";

  // ============================================================================
  // USM MEMORY ALLOCATION
  // ============================================================================
  int *A = malloc_shared<int>(N * N, q);
  int *B = malloc_shared<int>(N * N, q);
  int *C = malloc_shared<int>(N * N, q);

  // Initialize A & B with random values
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++) {
      A[i * N + j] = rand() % 10;
      B[i * N + j] = rand() % 10;
    }

  // ============================================================================
  // MATRIX MULTIPLICATION KERNEL
  // ============================================================================
  q.submit([&](handler &h) {
    h.parallel_for<class MatMulKernel>(range<2>(N, N), [=](id<2> idx) {
      size_t row = idx[0];
      size_t col = idx[1];

      int sum = 0;

      for (size_t k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];

      C[row * N + col] = sum;
    });
  });

  q.wait(); // Synchronize

  // ============================================================================
  // PRINT MATRICES
  // ============================================================================

  // Print A
  cout << "Matrix A:\n";
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++)
      cout << setw(4) << A[i * N + j];
    cout << "\n";
  }
  cout << "\n";

  // Print B
  cout << "Matrix B:\n";
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++)
      cout << setw(4) << B[i * N + j];
    cout << "\n";
  }
  cout << "\n";

  // Print C
  cout << "Matrix C = A × B:\n";
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++)
      cout << setw(6) << C[i * N + j];
    cout << "\n";
  }

  // Free USM memory
  free(A, q);
  free(B, q);
  free(C, q);

  return 0;
}

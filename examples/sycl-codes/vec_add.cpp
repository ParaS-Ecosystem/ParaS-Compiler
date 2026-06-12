// ================================================================
//                 VECTOR ADDITION USING SYCL (USM)
// ================================================================

#include <cstdlib>
#include <ctime>
#include <iomanip> // for setw()
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {

  size_t N;
  cout << "Enter size of vectors: ";
  cin >> N;

  srand(time(nullptr));

  queue q{cpu_selector_v};
  cout << "\nSYCL Device: " << q.get_device().get_info<info::device::name>()
       << "\n\n";

  int *A = malloc_shared<int>(N, q);
  int *B = malloc_shared<int>(N, q);
  int *C = malloc_shared<int>(N, q);

  for (size_t i = 0; i < N; i++) {
    A[i] = rand() % 100;
    B[i] = rand() % 100;
  }

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class vec_add>(
        range<1>(N), [=](id<1> idx) { C[idx] = A[idx] + B[idx]; });
  });
  q.wait();

  // ===================== ALIGNED TABLE OUTPUT ======================
  cout << "====================== VECTOR ADDITION RESULT "
          "======================\n\n";

  cout << left << setw(7) << "Index" << setw(10) << "A[i]" << setw(10) << "B[i]"
       << setw(15) << "C[i] = A+B"
       << "\n";

  cout << "--------------------------------------------------------------\n";

  for (size_t i = 0; i < N; i++) {
    cout << left << setw(7) << i << setw(10) << A[i] << setw(10) << B[i]
         << setw(15) << C[i] << "\n";
  }

  cout << "--------------------------------------------------------------\n";

  free(A, q);
  free(B, q);
  free(C, q);

  return 0;
}

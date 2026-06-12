#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;
using namespace std;

int main() {

  // -------------------------------
  // USER INPUT
  // -------------------------------
  size_t width, height;
  cout << "Enter image width  : ";
  cin >> width;
  cout << "Enter image height : ";
  cin >> height;

  size_t N = width * height;

  // -------------------------------
  // SYCL QUEUE + DEVICE INFO
  // -------------------------------
  queue q;
  cout << "\nSYCL Device: " << q.get_device().get_info<info::device::name>()
       << "\n\n";

  // -------------------------------
  // USM ALLOCATION
  // -------------------------------
  int *image = malloc_shared<int>(N, q);
  int *result = malloc_shared<int>(N, q);

  // Kernel is always 3×3 = 9 values
  int *kernel = malloc_shared<int>(9, q);

  // -------------------------------
  // INITIALIZE IMAGE
  // -------------------------------
  for (size_t i = 0; i < N; ++i)
    image[i] = i + 1;

  // Sobel kernel
  int tmpKernel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

  for (int i = 0; i < 9; i++)
    kernel[i] = tmpKernel[i];

  // Initialize result
  for (size_t i = 0; i < N; ++i)
    result[i] = 0;

  // -------------------------------
  // KERNEL SUBMISSION (USM)
  // -------------------------------
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class convolution>(
        range<2>(height, width), [=](item<2> id) {
          int x = id[0];
          int y = id[1];

          int sum = 0;

          // Apply 3×3 convolution
          for (int kx = -1; kx <= 1; ++kx) {
            for (int ky = -1; ky <= 1; ++ky) {

              int ix = x + kx;
              int iy = y + ky;

              if (ix >= 0 && ix < height && iy >= 0 && iy < width) {

                sum += image[ix * width + iy] * kernel[(kx + 1) * 3 + (ky + 1)];
              }
            }
          }

          result[x * width + y] = sum;
        });
  });

  q.wait();

  // -------------------------------
  // PRINT INPUT IMAGE
  // -------------------------------
  cout << "Input Image:\n";
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j)
      cout << setw(4) << image[i * width + j];
    cout << "\n";
  }
  cout << "\n";

  // -------------------------------
  // PRINT KERNEL
  // -------------------------------
  cout << "Kernel (3x3):\n";
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j)
      cout << setw(4) << kernel[i * 3 + j];
    cout << "\n";
  }
  cout << "\n";

  // -------------------------------
  // PRINT RESULT
  // -------------------------------
  cout << "Convolution Result:\n";
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j)
      cout << setw(6) << result[i * width + j];
    cout << "\n";
  }
  cout << "\n";

  // -------------------------------
  // FREE USM MEMORY
  // -------------------------------
  free(image, q);
  free(kernel, q);
  free(result, q);

  return 0;
}

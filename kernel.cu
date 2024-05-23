
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <cassert>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// define the size of the image
// assume square
#define ROWS 1024
#define COLS 1024

// constant for GPU mem
#define FILTER_RADIUS 2
__constant__ float FILTER[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];


__global__ void blurImage(int* input, int* output, float* filter, int radius, size_t width, size_t height) {
  // by using a global index to the total size of the array
  // we get the position by using the threadIDx, blockIdx* block Dimx 
  // same goes for y index
  int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
  float res = 0;
    for (int filterRow = 0; filterRow < 2*radius+1; filterRow++) {
      for (int filterCol = 0; filterCol < 2 * radius + 1; filterCol++) {
        int inputRow = y_idx - radius + filterRow;
        int inputCol = x_idx - radius + filterCol;
        if (inputRow >= 0 && inputRow < height && inputCol >=0 && inputCol < width){
          res += filter[(filterRow * 2 * radius + 1) + filterCol] * input[inputRow * width + inputCol];
      }
    }
  }
  output[y_idx*width+x_idx] = (int)res;
}

/* PGM format spec:
  * https://users.wpi.edu/~cfurlong/me-593n/pgmimage.html
  * header starts with P5 or P2 (ASCII)
  * width height
  * max_val
  * raw_data
  */
bool writePGM(const std::string& filename, int* data, size_t rows, size_t cols) {
  std::ofstream file(filename, std::ios::out);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  // PGM header
  // ASCII
  file << "P2" << std::endl;
  file << "# " << "test.pgm" << std::endl;
  file << cols << " " << rows << std::endl;
  // 8 bit
  file << 255 << std::endl;

  // Write pixel values
  for (size_t x = 0; x < rows; x++) {
    for (size_t y = 0; y < cols; y++) {
      file << data[x * cols + y] << " ";
    }
    file << std::endl;
  }

  file.close();
  return true;
}

void drawDiag(int* data, int rows, int cols, int width) {
  for (int i = 0; i < rows; ++i) {
    // Calculate the range of columns to modify for the current row
    int start_col = std::max(0, i - width);
    int end_col = std::min(cols - 1, i + width);

    for (int j = start_col; j <= end_col; ++j) {
      // Set the value at the corresponding 1D array index
      data[i * cols + j] = 0;
    }
  }
}

int main() {
  // creates white image
  int* data = new int[ROWS * COLS]{ 255 };

  drawDiag(data, ROWS, COLS, 2);
  bool img_file_created = writePGM("test.pgm", data, ROWS, COLS);
  assertm(img_file_created == true, "Image file created\n!");

  int* h_input = new int[ROWS * COLS];
  int* h_output = new int[ROWS * COLS];

  int filter_byte_size = sizeof(float) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);
  int input_byte_size = sizeof(int) * ROWS * COLS;

  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      h_input[i * COLS + j] = data[i * COLS + j];
    }
  }

  // initialize normalizecd filter 
  float* h_filter = new float[(2 * FILTER_RADIUS + 1)*(2 * FILTER_RADIUS + 1)];
  
  for (size_t i = 0; i < (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1); i++) {
    h_filter[i] = static_cast<float>(1.0f / 9.0f);
  }


  // GPU arrs
  int* d_input;
  int* d_output;

  // gpu alloc
  cudaMalloc(&d_input, input_byte_size);
  cudaMalloc(&d_output, input_byte_size);
  // inform data will NOT be changed
  cudaMemcpyToSymbol(FILTER, h_filter, filter_byte_size);

  // Copy data from the host to the device
  cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice);
  // might not need this
  //cudaMemcpy(FILTER, h_filter, filter_byte_size, cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 blockDim(16, 16);  // 16x16 threads per block
  dim3 gridDim((COLS + blockDim.x - 1) / blockDim.x, (ROWS + blockDim.y - 1) / blockDim.y);

  // Launch the kernel
  blurImage <<<gridDim, blockDim>>>(d_input, d_output, h_filter, FILTER_RADIUS, COLS, ROWS);

  // Copy the result back to the host
  cudaMemcpy(h_output, d_output, input_byte_size, cudaMemcpyDeviceToHost);

  bool blurred_img = writePGM("blur.pgm", h_output, ROWS, COLS);
  assertm(blurred_img == true, "Image file is blurred!\n");

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  delete[] h_input;
  delete[] h_output;
  delete[] h_filter;
  return 0;
}
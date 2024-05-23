
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <cassert>
#include <cmath>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// define the size of the image
// assume square
#define ROWS 1024
#define COLS 1024


#define FILTER_SIZE 3
#define FILTER_OFFSET (FILTER_SIZE/2);

// constant for GPU mem
__constant__ float FILTER[3 * 3];

__global__ void blurImage(float* input, float* output, float* filter, size_t width, size_t height) {
  // by using a global index to the total size of the array
  // we get the position by using the threadIDx, blockIdx* block Dimx 
  // same goes for y index
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  size_t N = width;

  int row_offset = row - FILTER_OFFSET;
  int col_offset = col - FILTER_OFFSET;
  
  float res = 0.0f;
  // Iterate over all the rows
  for (int i = 0; i < FILTER_SIZE; i++) {
    // Go over each column
    for (int j = 0; j < FILTER_SIZE; j++) {
      if ((row_offset + i) >= 0 && (row_offset + i) < N) {
        if ((col_offset + j) >= 0 && (col_offset + j) < N) {
          // Accumulate result
          res += input[(row_offset + i) * N + (col_offset + j)] *
            FILTER[i * FILTER_SIZE + j];
        }
      }
    }
  }
  // Write back the result
  output[row * N + col] = res;
}

/* PGM format spec:
  * https://users.wpi.edu/~cfurlong/me-593n/pgmimage.html
  * header starts with P5 or P2 (ASCII)
  * width height
  * max_val
  * raw_data
  */
bool writePGM(const std::string& filename, float* data, size_t rows, size_t cols) {
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
      file << std::round(data[x * cols + y]) << " ";
    }
    file << std::endl;
  }

  file.close();
  return true;
}

void drawDiag(float* data, int rows, int cols, int width) {
  for (int i = 0; i < rows; ++i) {
    // Calculate the range of columns to modify for the current row
    int start_col = std::max(0, i - width);
    int end_col = std::min(cols - 1, i + width);

    for (int j = start_col; j <= end_col; ++j) {
      // Set the value at the corresponding 1D array index
      data[i * cols + j] = 1;
    }
  }
}


int main() {
  int THREADS = 16;
  // creates white image
  float* h_input = new float[ROWS * COLS];
  std::fill(h_input, h_input + ROWS * COLS, 255.0f);
  drawDiag(h_input, ROWS, COLS, 2);

  bool img_file_created = writePGM("test.pgm", h_input, ROWS, COLS);
  assertm(img_file_created == true, "Image file created\n!");
  
  float* h_output = new float[ROWS * COLS];

  std::fill(h_output, h_output + ROWS * COLS, 0.0f);


  int filter_byte_size = sizeof(float) * FILTER_SIZE * FILTER_SIZE;
  int input_byte_size = sizeof(float) * ROWS * COLS;

  float* h_filter = new float[FILTER_SIZE * FILTER_SIZE];

  // initialize normalizecd filter 
  for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
    h_filter[i] = 1.0f / 9.0f;
  }

  // GPU arrs
  float* d_input;
  float* d_output;

  // gpu alloc
  cudaMalloc(&d_input, input_byte_size);
  cudaMalloc(&d_output, input_byte_size);
  // inform data will NOT be changed
  cudaMemcpyToSymbol(FILTER, h_filter, filter_byte_size);

  // Copy data from the host to the device
  cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, input_byte_size, cudaMemcpyHostToDevice);
  // might not need this
  //cudaMemcpy(FILTER, h_filter, filter_byte_size, cudaMemcpyHostToDevice);

  int block_size = (1024 + THREADS - 1) / THREADS;

  // Define block and grid dimensions
  dim3 blockDim(THREADS, THREADS);  // 16x16 threads per block
  dim3 gridDim(block_size, block_size);

  // Launch the kernel
  blurImage <<<gridDim, blockDim>>>(d_input, d_output, FILTER, ROWS, COLS);

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
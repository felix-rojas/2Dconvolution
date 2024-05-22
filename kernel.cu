
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

// define the size of the image
#define ROWS 1024
#define COLS 1024

// define threads per block 
#define HILOS_POR_BLOQUE 512

__global__ void blurImage(float* input, float* output, size_t width, size_t height) {
  // by using a global index to the total size of the array
  // we get the position by using the threadIDx, blockIdx* block Dimx 
  // same goes for y index
  size_t x_idx = threadIdx.x + blockIdx.x * blockDim.x;

  // maybe dim 2 index? size_t y_idx = threadIdx.y + blockIdx.y * blockDim.y;
  // the number of threads might not match so we need to guard that limit
  if (x_idx < width) {
    float sum = 0;
    size_t count = 0;
  }
}

bool make_img() {
  /* PGM format spec:
  * https://users.wpi.edu/~cfurlong/me-593n/pgmimage.html
  * header starts with P5
  * width height
  * max_val
  * raw_data
  */
  const char* filename = "black_diagonal.pgm";
  /* 2D img array, no color */
  unsigned char data[COLS][ROWS];

  // bitshift value format
  const int bit_format = 8;
  const int max_val = (1 << bit_format) - 1;

  FILE* fp;
  const char* comment = "#Diagonal black line";

  /* black diagonal */
  for (size_t y = 0; y < COLS; ++y) {
    for (size_t x = 0; x < ROWS; ++x) {
      if (x == y) {
        data[y][x] = 0; // black
      }
      else {
        data[y][x] = max_val; // white
      }
    }
  }

  /* write the whole data array to ppm file in one step */
  /* create new file, give it a name and open it in binary mode */
  fp = fopen(filename, "wb");
  /* write header to the file */
  fprintf(fp, "P5\n %s\n %d\n %d\n %d\n", comment, COLS, ROWS,
    max_val);
  /* write image data bytes to the file */
  fwrite(data, sizeof(data), 1, fp);
  fclose(fp);

  return true;
}


int main() {
  bool img_file_created = make_img();
  if (img_file_created) {
    int* h_a, * h_b, * h_c;
    int* d_a, * d_b, * d_c;

    //size_t arr_size = ROWS * sizeof(float);

    cudaMalloc((void**)&d_a, arr_size);
    cudaMalloc((void**)&d_b, arr_size);
    cudaMalloc((void**)&d_c, arr_size);

    h_a = (int*)malloc(arr_size);
    h_b = (int*)malloc(arr_size);
    h_c = (int*)malloc(arr_size);


    cudaMemcpy(d_a, h_a, arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arr_size, cudaMemcpyHostToDevice);


    //blurImage << < N / HILOS_POR_BLOQUE, HILOS_POR_BLOQUE >> > (d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, arr_size, cudaMemcpyDeviceToHost);


    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
  }
  else {
    printf("Could not create file");
    return 0;
  }
}
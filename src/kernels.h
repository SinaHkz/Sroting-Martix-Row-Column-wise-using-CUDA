#include "stdlib.h"
#include "utilities.h"


#include <cuda_runtime.h>

// Declare the kernel
__global__ void sortRowsKernelInt(int *matrix, int rows, int cols);

__global__ void sortRowsKernelFloat(float *matrix, int rows, int cols);

__global__ void sortRowsKernelDouble(double *matrix, int rows, int cols);

__global__ void transposeKernelInt(int *input, int *output, int rows, int cols) ;

__global__ void transposeKernelDouble(double *input, double *output, int rows, int cols);

__global__ void transposeKernelFloat(float *input, float *output, int rows, int cols) ;

// __device__ void mergeSortInt(int* arr, int left, int right);

// __device__ void mergeSortFloat(float* arr, int left, int right);

// __device__ void mergeSortDouble(double* arr, int left, int right);

__device__ void bubbleSortDouble(double *row, int cols);
__device__ void bubbleSortFloat(float *row, int cols);
__device__ void bubbleSortInt(int *row, int cols);
#include "stdlib.h"
// #include "utilities.h"



__global__ void checkSortedRowWiseInt(int *matrix, int rows, int cols, int *foundUnsorted);

__global__ void checkSortedRowWiseFloat(float *matrix, int rows, int cols, int *foundUnsorted);

__global__ void checkSortedRowWiseDouble(double *matrix, int rows, int cols, int *foundUnsorted);

__global__ void checkSortedColumnWiseInt(int *matrix, int rows, int cols, int *foundUnsorted);

__global__ void checkSortedColumnWiseFloat(float *matrix, int rows, int cols, int *foundUnsorted);

__global__ void checkSortedColumnWiseDouble(double *matrix, int rows, int cols, int *foundUnsorted);

// Declare the kernel
__global__ void sortRowsKernelInt(int *matrix, int rows, int cols);

__global__ void sortRowsKernelFloat(float *matrix, int rows, int cols);

__global__ void sortRowsKernelDouble(double *matrix, int rows, int cols);

__global__ void transposeKernelInt(int *input, int *output, int rows, int cols) ;

__global__ void transposeKernelDouble(double *input, double *output, int rows, int cols);

__global__ void transposeKernelFloat(float *input, float *output, int rows, int cols) ;


__device__ void bubbleSortDouble(double *row, int cols);
__device__ void bubbleSortFloat(float *row, int cols);
__device__ void bubbleSortInt(int *row, int cols);


__device__ void mergeSortInt(int *sharedRowInt, int cols);
__device__ void mergeSortFloat(float *sharedRow, int cols);
__device__ void mergeSortDouble(double *sharedRow, int cols);

__device__ void mergeInt(int *sharedRowInt, int left, int mid, int right);
__device__ void mergeFloat(float *sharedRow, int left, int mid, int right);
__device__ void mergeDouble(double *sharedRow, int left, int mid, int right);
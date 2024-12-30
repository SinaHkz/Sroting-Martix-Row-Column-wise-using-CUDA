#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../includes/common.h"


void* readMatrix(const char* filename, int* rows, int* cols, MatrixType* type);

void writeMatrix(const char* filename, void* matrix, int rows, int cols, MatrixType type);

bool isMatrixSorted(void *matrix, int rows, int cols, MatrixType type) ;

// __device__ void mergeSortInt(int* arr, int left, int right);

// __device__ void mergeSortFloat(float* arr, int left, int right);

// __device__ void mergeSortDouble(double* arr, int left, int right);

#include "kernels.h"



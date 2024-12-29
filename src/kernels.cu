#include "kernels.h"

#define TILE_SIZE 32



// kernels to sort all rows with multiple data types
__global__ void sortRowsKernelInt(int **matrix, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
    {
        bubbleSortInt(matrix[i], 0, cols);
    }
}

__global__ void sortRowsKernelFloat(float **matrix, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
    {
        bubbleSortFloat(matrix[i], 0, cols);
    }
}

__global__ void sortRowsKernelDouble(double **matrix, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows)
    {
        bubbleSortDouble(matrix[i], 0, cols);
    }
}



// kernels to transpose matrix for multiple data types
__global__ void transposeKernelInt(int **input, int *output, int rows, int cols) {
    __shared__ int tile[TILE_SIZE][TILE_SIZE + 1];  // Shared memory with padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;  // Global column index
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;  // Global row index

    int local_x = threadIdx.x;  // Local column index in the tile
    int local_y = threadIdx.y;  // Local row index in the tile

    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[local_y][local_x] = input[y][x];
    }

    __syncthreads();  // Ensure all threads in the block finish loading data

    // Transpose indices
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed data to global memory
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[local_x][local_y];
    }
}

__global__ void transposeKernelDouble(const double *input, double *output, int rows, int cols, int tileSize) {
    __shared__ double tile[TILE_SIZE][TILE_SIZE + 1];  // Shared memory with padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;  // Global column index
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;  // Global row index

    int local_x = threadIdx.x;  // Local column index in the tile
    int local_y = threadIdx.y;  // Local row index in the tile

    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[local_y][local_x] = input[y * cols + x];
    }

    __syncthreads();  // Ensure all threads in the block finish loading data

    // Transpose indices
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed data to global memory
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[local_x][local_y];
    }
}

__global__ void transposeKernelFloat(float *a, float *b, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // Shared memory with padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;  // Global column index
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;  // Global row index

    // Read from global memory to shared memory (if within bounds)
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = a[y * cols + x];

    __syncthreads();  // Wait for all threads to finish loading

    // Transpose within shared memory
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // Transposed global column index
    y = blockIdx.x * TILE_SIZE + threadIdx.y;  // Transposed global row index

    // Write transposed data back to global memory (if within bounds)
    if (x < rows && y < cols)
        b[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}




//sorting functions for multiple data types
__device__ void swapInt(int *arr, int i, int j)
{
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

__device__ void swapFloat(float *arr, int i, int j)
{
    float temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

__device__ void swapDouble(double *arr, int i, int j)
{
    double temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

__device__ void bubbleSortInt(int *arr, int left, int right)
{
    bool swapped;
    for (int i = left; i < right - 1; i++)  // outer loop for passes
    {
        swapped = false;
        for (int j = left; j < right - 1 - i; j++)  // inner loop for comparing adjacent elements
        {
            if (arr[j] > arr[j + 1])
            {
                swapInt(arr, j, j + 1);  // swap elements if they are in the wrong order
                swapped = true;
            }
        }

        // If no elements were swapped in this pass, the array is already sorted
        if (!swapped)
            break;
    }
}

__device__ void bubbleSortFloat(float *arr, int left, int right)
{
    bool swapped;
    for (int i = left; i < right - 1; i++)  // outer loop for passes
    {
        swapped = false;
        for (int j = left; j < right - 1 - i; j++)  // inner loop for comparing adjacent elements
        {
            if (arr[j] > arr[j + 1])
            {
                swapFloat(arr, j, j + 1);  // swap elements if they are in the wrong order
                swapped = true;
            }
        }

        // If no elements were swapped in this pass, the array is already sorted
        if (!swapped)
            break;
    }
}

__device__ void bubbleSortDouble(double *arr, int left, int right)
{
    bool swapped;
    for (int i = left; i < right - 1; i++)  // outer loop for passes
    {
        swapped = false;
        for (int j = left; j < right - 1 - i; j++)  // inner loop for comparing adjacent elements
        {
            if (arr[j] > arr[j + 1])
            {
                swapDouble(arr, j, j + 1);  // swap elements if they are in the wrong order
                swapped = true;
            }
        }

        // If no elements were swapped in this pass, the array is already sorted
        if (!swapped)
            break;
    }
}

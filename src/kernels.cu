#include "kernels.h"

#define TILE_SIZE 32
#define BLOCK_ROWS 128

// kernel to check if a matrix is sorted or not
__global__ void checkSortedRowWiseInt(int *matrix, int rows, int cols, int *foundUnsorted)
{
    extern __shared__ int sharedRowCheckInt[];
    if (*foundUnsorted)
        return;
    int idx = blockIdx.x;
    if (idx < rows)
    {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            sharedRowCheckInt[i] = matrix[idx * cols + i];

        __syncthreads();
        // Check if an unsorted element has already been found (early exit)

        for (int j = threadIdx.x; j * 2 < cols - 1; j += blockDim.x)
        {
            if (sharedRowCheckInt[j * 2] > sharedRowCheckInt[j * 2 + 1])
            {
                atomicExch(foundUnsorted, 1);
                return;
            }
        }

        for (int j = threadIdx.x; j * 2 < cols - 2; j += blockDim.x)
        {
            if (sharedRowCheckInt[j * 2 + 1] > sharedRowCheckInt[j * 2 + 2])
            {
                atomicExch(foundUnsorted, 1);
                return;
            }
        }

        if (*foundUnsorted)
            return;
    }
}

__global__ void checkSortedRowWiseFloat(float *matrix, int rows, int cols, int *foundUnsorted)
{
    extern __shared__ int sharedRowCheckFloat[];
    if (*foundUnsorted)
        return;
    int idx = blockIdx.x;
    if (idx < rows)
    {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            sharedRowCheckFloat[i] = matrix[idx * cols + i];

        __syncthreads();
        // Check if an unsorted element has already been found (early exit)

        for (int j = threadIdx.x; j * 2 < cols - 1; j += blockDim.x)
        {
            if (sharedRowCheckFloat[j * 2] > sharedRowCheckFloat[j * 2 + 1])
            {
                atomicExch(foundUnsorted, 1);
                return;
            }
        }

        for (int j = threadIdx.x; j * 2 < cols - 2; j += blockDim.x)
        {
            if (sharedRowCheckFloat[j * 2 + 1] > sharedRowCheckFloat[j * 2 + 2])
            {
                atomicExch(foundUnsorted, 1);
                return;
            }
        }

        if (*foundUnsorted)
            return;
    }
}

__global__ void checkSortedRowWiseDouble(double *matrix, int rows, int cols, int *foundUnsorted)
{
    extern __shared__ int sharedRowCheckDouble[];
    if (*foundUnsorted)
        return;
    int idx = blockIdx.x;
    if (idx < rows)
    {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            sharedRowCheckDouble[i] = matrix[idx * cols + i];

        __syncthreads();
        // Check if an unsorted element has already been found (early exit)

        for (int j = threadIdx.x; j * 2 < cols - 1; j += blockDim.x)
        {
            if (sharedRowCheckDouble[j * 2] > sharedRowCheckDouble[j * 2 + 1])
            {
                atomicExch(foundUnsorted, 1);
                return;
            }
        }

        for (int j = threadIdx.x; j * 2 < cols - 2; j += blockDim.x)
        {
            if (sharedRowCheckDouble[j * 2 + 1] > sharedRowCheckDouble[j * 2 + 2])
            {
                atomicExch(foundUnsorted, 1);
                return;
            }
        }

        if (*foundUnsorted)
            return;
    }
}

// kernels to sort all rows with multiple data types
__global__ void sortRowsKernelInt(int *matrix, int rows, int cols)
{
    extern __shared__ int sharedRowInt[];

    int row = blockIdx.x;
    if (row < rows)
    {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            sharedRowInt[i] = matrix[row * cols + i];

        __syncthreads();

       mergeSortInt(sharedRowInt, cols);
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            matrix[row * cols + i] = sharedRowInt[i];
    }
}

__global__ void sortRowsKernelFloat(float *matrix, int rows, int cols)
{
    extern __shared__ float sharedRowFloat[];

    int row = blockIdx.x;
    if (row < rows)
    {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            sharedRowFloat[i] = matrix[row * cols + i];

        __syncthreads();

        bubbleSortFloat(sharedRowFloat, cols);
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            matrix[row * cols + i] = sharedRowFloat[i];
    }
}

__global__ void sortRowsKernelDouble(double *matrix, int rows, int cols)
{
    extern __shared__ double sharedRowDouble[];

    int row = blockIdx.x;
    if (row < rows)
    {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            sharedRowDouble[i] = matrix[row * cols + i];

        __syncthreads();

        mergeSortDouble(sharedRowDouble, cols);
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            matrix[row * cols + i] = sharedRowDouble[i];
    }
}

// kernels to transpose matrix for multiple data types
__global__ void transposeKernelInt(int *input, int *output, int rows, int cols)
{
    __shared__ int tile[TILE_SIZE][TILE_SIZE + 1]; // Shared memory without padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x; // Global column index
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; // Global row index

    int local_x = threadIdx.x; // Local column index in the tile
    int local_y = threadIdx.y; // Local row index in the tile

    // Load data into shared memory with bounds checking
    if (x < cols && y < rows && local_x < TILE_SIZE && local_y < TILE_SIZE)
    {
        tile[local_y][local_x] = input[y * cols + x];
    }

    __syncthreads(); // Ensure all threads in the block finish loading data

    // Transpose indices
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed data to global memory with bounds checking
    if (x < rows && y < cols && local_x < TILE_SIZE && local_y < TILE_SIZE)
    {
        output[y * rows + x] = tile[local_x][local_y];
    }
}

__global__ void transposeKernelDouble(double *input, double *output, int rows, int cols)
{
    __shared__ double tile[TILE_SIZE][TILE_SIZE + 1]; // Shared memory with padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x; // Global column index
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; // Global row index

    int local_x = threadIdx.x; // Local column index in the tile
    int local_y = threadIdx.y; // Local row index in the tile

    // Load data into shared memory
    if (x < cols && y < rows)
    {
        tile[local_y][local_x] = input[y * cols + x];
    }

    __syncthreads(); // Ensure all threads in the block finish loading data

    // Transpose indices
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed data to global memory
    if (x < rows && y < cols)
    {
        output[y * rows + x] = tile[local_x][local_y];
    }
}

__global__ void transposeKernelFloat(float *input, float *output, int rows, int cols)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // Shared memory with padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x; // Global column index
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; // Global row index

    // Read from global memory to shared memory (if within bounds)
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    __syncthreads(); // Wait for all threads to finish loading

    // Transpose within shared memory
    x = blockIdx.y * TILE_SIZE + threadIdx.x; // Transposed global column index
    y = blockIdx.x * TILE_SIZE + threadIdx.y; // Transposed global row index

    // Write transposed data back to global memory (if within bounds)
    if (x < rows && y < cols)
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

// sorting functions for multiple data types
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

__device__ void bubbleSortInt(int *row, int cols)
{
    for (int i = 0; i < cols - 1; i++)
    {
        for (int j = 0; j < cols - i - 1; j++)
        {
            if (row[j] > row[j + 1])
            {
                // Swap the elements
                int temp = row[j];
                row[j] = row[j + 1];
                row[j + 1] = temp;
            }
        }
    }
}

__device__ void bubbleSortFloat(float *row, int cols)
{
    for (int i = 0; i < cols - 1; i++)
    {
        for (int j = 0; j < cols - i - 1; j++)
        {
            if (row[j] > row[j + 1])
            {
                // Swap the elements
                float temp = row[j];
                row[j] = row[j + 1];
                row[j + 1] = temp;
            }
        }
    }
}

__device__ void bubbleSortDouble(double *row, int cols)
{
    for (int i = 0; i < cols - 1; i++)
    {
        for (int j = 0; j < cols - i - 1; j++)
        {
            if (row[j] > row[j + 1])
            {
                // Swap the elements
                double temp = row[j];
                row[j] = row[j + 1];
                row[j + 1] = temp;
            }
        }
    }
}

__device__ void mergeInt(int *sharedRowInt, int left, int mid, int right)
{
    int i = left;    // Pointer for the left half
    int j = mid;     // Pointer for the right half

    // Merge the two halves in-place
    while (i < mid && j < right)
    {
        if (sharedRowInt[i] <= sharedRowInt[j])
        {
            i++;
        }
        else
        {
            int temp = sharedRowInt[j];
            // Shift elements to the right
            for (int m = j; m > i; m--)
            {
                sharedRowInt[m] = sharedRowInt[m - 1]; // Shift right
            }
            sharedRowInt[i] = temp; // Insert the element in the correct position
            i++;
            j++;
        }
    }
}

__device__ void mergeSortInt(int *sharedRowInt, int cols)
{
    // Iterative bottom-up merge sort
    for (int size = 1; size < cols; size *= 2)
    {
        for (int left = 0; left < cols; left += 2 * size)
        {
            int mid = min(left + size, cols);
            int right = min(left + 2 * size, cols);

            // Merge the two sorted halves
            mergeInt(sharedRowInt, left, mid, right);
        }
        __syncthreads();
    }
}

__device__ void mergeFloat(float *sharedRow, int left, int mid, int right)
{
    int i = left;    // Pointer for the left half
    int j = mid;     // Pointer for the right half

    // Merge the two halves in-place
    while (i < mid && j < right)
    {
        if (sharedRow[i] <= sharedRow[j])
        {
            i++;
        }
        else
        {
            float temp = sharedRow[j];
            // Shift elements to the right
            for (int m = j; m > i; m--)
            {
                sharedRow[m] = sharedRow[m - 1]; // Shift right
            }
            sharedRow[i] = temp; // Insert the element in the correct position
            i++;
            j++;
        }
    }
}

__device__ void mergeSortFloat(float *sharedRow, int cols)
{
    // Iterative bottom-up merge sort
    for (int size = 1; size < cols; size *= 2)
    {
        for (int left = 0; left < cols; left += 2 * size)
        {
            int mid = min(left + size, cols);
            int right = min(left + 2 * size, cols);

            // Merge the two sorted halves
            mergeFloat(sharedRow, left, mid, right);
        }
        __syncthreads();
    }
}

__device__ void mergeDouble(double *sharedRow, int left, int mid, int right)
{
    int i = left;    // Pointer for the left half
    int j = mid;     // Pointer for the right half

    // Merge the two halves in-place
    while (i < mid && j < right)
    {
        if (sharedRow[i] <= sharedRow[j])
        {
            i++;
        }
        else
        {
            double temp = sharedRow[j];
            // Shift elements to the right
            for (int m = j; m > i; m--)
            {
                sharedRow[m] = sharedRow[m - 1]; // Shift right
            }
            sharedRow[i] = temp; // Insert the element in the correct position
            i++;
            j++;
        }
    }
}

__device__ void mergeSortDouble(double *sharedRow, int cols)
{
    // Iterative bottom-up merge sort
    for (int size = 1; size < cols; size *= 2)
    {
        for (int left = 0; left < cols; left += 2 * size)
        {
            int mid = min(left + size, cols);
            int right = min(left + 2 * size, cols);

            // Merge the two sorted halves
            mergeDouble(sharedRow, left, mid, right);
        }
        __syncthreads();
    }
}
#include "kernels.h"

#define TILE_SIZE 32

// kernels to sort all rows with multiple data types
__global__ void sortRowsKernelInt(int *matrix, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
    {
        int start = i * cols;                // Calculate the starting index for this row
        bubbleSortInt(matrix + start, cols); // Sorting the row using the start index, 1 row, and cols columns
    }
}

__global__ void sortRowsKernelFloat(float *matrix, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
    {
        int start = i * cols;                  // Calculate the starting index for this row
        bubbleSortFloat(matrix + start, cols); // Sorting the row using the start index, 1 row, and cols columns
    }
}

__global__ void sortRowsKernelDouble(double *matrix, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
    {
        int start = i * cols;                   // Calculate the starting index for this row
        bubbleSortDouble(matrix + start, cols); // Sorting the row using the start index, 1 row, and cols columns
    }
}

// kernels to transpose matrix for multiple data types
__global__ void transposeKernelInt(int *input, int *output, int rows, int cols)
{
    __shared__ int tile[TILE_SIZE][TILE_SIZE + 1]; // Shared memory with padding

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


__device__ void mergeSortInt(int *array, int cols)
{
    int *temp = (int *)malloc(cols * sizeof(int)); // Temporary array for merging

    for (int width = 1; width < cols; width *= 2) // Increase width of subarrays
    {
        for (int i = 0; i < cols; i += 2 * width) // Iterate over pairs of subarrays
        {
            int left = i;                 // Left subarray start
            int mid = min(i + width, cols); // End of left subarray (start of right)
            int right = min(i + 2 * width, cols); // End of right subarray

            int l = left, r = mid, k = left;

            // Merge the two subarrays
            while (l < mid && r < right)
            {
                if (array[l] <= array[r])
                {
                    temp[k++] = array[l++];
                }
                else
                {
                    temp[k++] = array[r++];
                }
            }

            // Copy remaining elements from left subarray
            while (l < mid)
            {
                temp[k++] = array[l++];
            }

            // Copy remaining elements from right subarray
            while (r < right)
            {
                temp[k++] = array[r++];
            }
        }

        // Copy merged subarray back to the original array
        for (int i = 0; i < cols; i++)
        {
            array[i] = temp[i];
        }
    }

    free(temp); // Free the temporary array
}

__device__ void mergeSortFloat(float *array, int cols)
{
    float *temp = (float *)malloc(cols * sizeof(float)); // Temporary array for merging

    for (int width = 1; width < cols; width *= 2) // Increase width of subarrays
    {
        for (int i = 0; i < cols; i += 2 * width) // Iterate over pairs of subarrays
        {
            int left = i;                 // Left subarray start
            int mid = min(i + width, cols); // End of left subarray (start of right)
            int right = min(i + 2 * width, cols); // End of right subarray

            int l = left, r = mid, k = left;

            // Merge the two subarrays
            while (l < mid && r < right)
            {
                if (array[l] <= array[r])
                {
                    temp[k++] = array[l++];
                }
                else
                {
                    temp[k++] = array[r++];
                }
            }

            // Copy remaining elements from left subarray
            while (l < mid)
            {
                temp[k++] = array[l++];
            }

            // Copy remaining elements from right subarray
            while (r < right)
            {
                temp[k++] = array[r++];
            }
        }

        // Copy merged subarray back to the original array
        for (int i = 0; i < cols; i++)
        {
            array[i] = temp[i];
        }
    }

    free(temp); // Free the temporary array
}

__device__ void mergeSortDouble(double *array, int cols)
{
    double *temp = (double *)malloc(cols * sizeof(double)); // Temporary array for merging

    for (int width = 1; width < cols; width *= 2) // Increase width of subarrays
    {
        for (int i = 0; i < cols; i += 2 * width) // Iterate over pairs of subarrays
        {
            int left = i;                 // Left subarray start
            int mid = min(i + width, cols); // End of left subarray (start of right)
            int right = min(i + 2 * width, cols); // End of right subarray

            int l = left, r = mid, k = left;

            // Merge the two subarrays
            while (l < mid && r < right)
            {
                if (array[l] <= array[r])
                {
                    temp[k++] = array[l++];
                }
                else
                {
                    temp[k++] = array[r++];
                }
            }

            // Copy remaining elements from left subarray
            while (l < mid)
            {
                temp[k++] = array[l++];
            }

            // Copy remaining elements from right subarray
            while (r < right)
            {
                temp[k++] = array[r++];
            }
        }

        // Copy merged subarray back to the original array
        for (int i = 0; i < cols; i++)
        {
            array[i] = temp[i];
        }
    }

    free(temp); // Free the temporary array
}

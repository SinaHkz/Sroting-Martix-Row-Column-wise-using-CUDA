#include "utilities.h"

#define BLOCKSIZE2D 32
#define BLOCKSIZE1D 32

void copyMatrixFromDeviceToHost(void *deviceMatrix, void *matrix, int rows, int cols, MatrixType type)
{
    size_t elementSize;
    if (type == INT)
    {
        elementSize = sizeof(int);
    }
    else if (type == FLOAT)
    {
        elementSize = sizeof(float);
    }
    else
    {
        elementSize = sizeof(double);
    }
    cudaMemcpy(matrix, deviceMatrix, rows * cols * elementSize, cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv)
{
    int rows, cols;
    MatrixType type;
    void *deviceMatrix;
    void *output;
    int *foundUnsorted;
    int sharedMemSize;

    // read the matrix from input file
    void *matrix = readMatrix(argv[1], &rows, &cols, &type);

    if (matrix == NULL)
        return 1;

    // set grid and block size
    dim3 block1D(BLOCKSIZE1D);
    dim3 grid1D(rows);

    dim3 block2D(BLOCKSIZE2D, BLOCKSIZE2D);
    dim3 grid2D((cols + BLOCKSIZE2D - 1) / BLOCKSIZE2D, (rows + BLOCKSIZE2D - 1) / BLOCKSIZE2D);

    // Allocate memory on the device
    size_t elementSize;
    if (type == INT)
    {
        elementSize = sizeof(int);
    }
    else if (type == FLOAT)
    {
        elementSize = sizeof(float);
    }
    else
    {
        elementSize = sizeof(double);
    }
    sharedMemSize = cols * elementSize;

    cudaMalloc(&deviceMatrix, rows * cols * elementSize);
    cudaMalloc(&output, rows * cols * elementSize);
    cudaMallocManaged(&foundUnsorted, sizeof(int));
    cudaMemcpy(deviceMatrix, matrix, rows * cols * elementSize, cudaMemcpyHostToDevice);

    // Launch kernels for sorting
    while (!isMatrixSorted(deviceMatrix, rows, cols, type, foundUnsorted))
    {
        int temp;
        switch (type)
        {
        case INT:
            sortRowsKernelInt<<<grid1D, block1D, sharedMemSize>>>((int *)deviceMatrix, rows, cols);
            transposeKernelInt<<<grid2D, block2D>>>((int *)deviceMatrix, (int *)output, rows, cols);
            temp = rows;
            rows = cols;
            cols = temp;
            break;
        case FLOAT:
            sortRowsKernelFloat<<<grid1D, block1D, sharedMemSize>>>((float *)deviceMatrix, rows, cols);
            transposeKernelFloat<<<grid2D, block2D>>>((float *)deviceMatrix, (float *)output, rows, cols);
            temp = rows;
            rows = cols;
            cols = temp;
            break;
        case DOUBLE:
            sortRowsKernelDouble<<<grid1D, block1D, sharedMemSize>>>((double *)deviceMatrix, rows, cols);
            transposeKernelDouble<<<grid2D, block2D>>>((double *)deviceMatrix, (double *)output, rows, cols);
            temp = rows;
            rows = cols;
            cols = temp;

            break;
        }
        cudaMemcpy(deviceMatrix, output, rows * cols * elementSize, cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();

    // Copy result back to host
    copyMatrixFromDeviceToHost(deviceMatrix, matrix, rows, cols, type);

    // Free device memory
    cudaFree(deviceMatrix);
    cudaFree(output);
    cudaFree(foundUnsorted);
    // Write the matrix to file
    writeMatrix(argv[2], matrix, rows, cols, type);

    return 0;
}

// for (int i = 0; i < rows; i++) {
//     for (int j = 0; j < cols; j++) {
//         if (type == INT)
//             printf("%d ", ((int *)deviceMatrix)[i * cols + j]);
//         else if (type == FLOAT)
//             printf("%f ", ((float *)deviceMatrix)[i * cols + j]);
//         else if (type == DOUBLE)
//             printf("%lf ", ((double *)deviceMatrix)[i * cols + j]);
//     }
//     printf("\n");
// }

// printf("\n");

// for (int i = 0; i < rows; i++) {
//     for (int j = 0; j < cols; j++) {
//         if (type == INT)
//             printf("%d ", ((int *)output)[i * rows + j]);
//         else if (type == FLOAT)
//             printf("%f ", ((float *)output)[i * rows + j]);
//         else if (type == DOUBLE)
//             printf("%lf ", ((double *)output)[i * rows + j]);
//     }
//     printf("\n");
// }
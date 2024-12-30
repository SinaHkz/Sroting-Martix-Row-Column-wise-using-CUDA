#include "kernels.h"

#define BLOCKSIZE 32

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

    // read the matrix from input file
    void *matrix = readMatrix(argv[1], &rows, &cols, &type);

    if (matrix == NULL)
        return 1;

    // set grid and block size
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((rows + block.x - 1) / block.x, (rows + block.x - 1) / block.x);

    void *deviceMatrix;

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

    cudaMalloc(&deviceMatrix, rows * cols * elementSize);
    cudaMemcpy(deviceMatrix, matrix, rows * cols * elementSize, cudaMemcpyHostToDevice);

    // void *output;
    // cudaMalloc(&output, rows * cols * elementSize); // Assuming output is for int transpose

    // Launch kernels for sorting
    for (int i = 0; i < 2; i++)
    {

        switch (type)
        {
        case INT:
            sortRowsKernelInt<<<grid, block>>>((int *)deviceMatrix, rows, cols);
            transposeKernelInt<<<grid, block>>>((int *)deviceMatrix, (int *)deviceMatrix, rows, cols);
            // deviceMatrix = output;
            break;
        case FLOAT:

            sortRowsKernelFloat<<<grid, block>>>((float *)deviceMatrix, rows, cols);
            transposeKernelFloat<<<grid, block>>>((float *)deviceMatrix, (float *)deviceMatrix, rows, cols);
            break;
        case DOUBLE:
            sortRowsKernelDouble<<<grid, block>>>((double *)deviceMatrix, rows, cols);
            transposeKernelDouble<<<grid, block>>>((double *)deviceMatrix, (double *)deviceMatrix, rows, cols);

            break;
        }
    }

    cudaDeviceSynchronize();

    // Copy result back to host
    copyMatrixFromDeviceToHost(deviceMatrix, matrix, rows, cols, type);

    // Free device memory
    cudaFree(deviceMatrix);
    // cudaFree(output);

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
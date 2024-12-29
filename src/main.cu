#include "kernels.h"

#define BLOCKSIZE 32

void copyMatrixFromDeviceToHost(void **deviceMatrix, void **matrix, int rows, int cols, MatrixType type)
{
    switch (type)
    {
    case INT:
        for (int i = 0; i < rows; i++)
        {
            cudaMemcpy(matrix[i], deviceMatrix[i], cols * sizeof(int), cudaMemcpyDeviceToHost);
        }
        break;

    case FLOAT:
        for (int i = 0; i < rows; i++)
        {
            cudaMemcpy(matrix[i], deviceMatrix[i], cols * sizeof(float), cudaMemcpyDeviceToHost);
        }
        break;

    case DOUBLE:
        for (int i = 0; i < rows; i++)
        {
            cudaMemcpy(matrix[i], deviceMatrix[i], cols * sizeof(double), cudaMemcpyDeviceToHost);
        }
        break;
    }
}

int main(int argc, char **argv)
{
    int rows, cols;
    MatrixType type;

    // read the matrix from input file
    void **matrix = readMatrix(argv[1], &rows, &cols, &type);

    if (matrix == NULL)
        return 1;

    // set grid and bloxk size
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((rows + block.x - 1) / block.x);

    void **deviceMatrix;

    switch (type)
    {
    case INT:
        cudaMallocManaged(&deviceMatrix, rows * sizeof(int *));
        for (int i = 0; i < rows; i++)
        {
            cudaMallocManaged(&(deviceMatrix[i]), cols * sizeof(int));
            cudaMemcpy(deviceMatrix[i], matrix[i], cols * sizeof(int), cudaMemcpyHostToDevice);
        }
        break;

    case FLOAT:
        cudaMallocManaged(&deviceMatrix, rows * sizeof(float *));
        for (int i = 0; i < rows; i++)
        {
            cudaMallocManaged(&(deviceMatrix[i]), cols * sizeof(float));
            cudaMemcpy(deviceMatrix[i], matrix[i], cols * sizeof(float), cudaMemcpyHostToDevice);
        }
        break;

    case DOUBLE:
        cudaMallocManaged(&deviceMatrix, rows * sizeof(double *));
        for (int i = 0; i < rows; i++)
        {
            cudaMallocManaged(&(deviceMatrix[i]), cols * sizeof(double));
            cudaMemcpy(deviceMatrix[i], matrix[i], cols * sizeof(double), cudaMemcpyHostToDevice);
        }
        break;
    }
    void* output;
    cudaMallocManaged(&output, rows * cols * sizeof(int *));
    

    // launchSortKernel(deviceMatrix, rows, cols, type, block, grid);
    switch (type)
    {
    case INT:
        sortRowsKernelInt<<<grid, block>>>((int **)(deviceMatrix), rows, cols);
        transposeKernelInt<<<grid, block>>>((int**)(deviceMatrix), (int*)output, rows, cols);
        break;
    case FLOAT:
        sortRowsKernelFloat<<<grid, block>>>((float **)(deviceMatrix), rows, cols);
        break;
    case DOUBLE:
        sortRowsKernelDouble<<<grid, block>>>((double **)(deviceMatrix), rows, cols);
        break;
    }


    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", ((int *)deviceMatrix[i])[j]);
        }
        printf("\n");
    }

    printf("\n");



    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", ((int *)output)[i * rows + j]);
        }
        printf("\n");
    }

    copyMatrixFromDeviceToHost(deviceMatrix, matrix, rows, cols, type);

    for (int i = 0; i < rows; i++)
    {
        cudaFree(deviceMatrix[i]);
    }
    cudaFree(deviceMatrix);

    writeMatrix(argv[2], matrix, rows, cols, type);

    return 0;
}


#include "utilities.h"

#define BLOCKSIZE2D 32
#define BLOCKSIZE1D 32
#define MAX_STREAMS 16

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

    // Create streams for overlapping
    cudaStream_t streams[MAX_STREAMS];
    for (int i = 0; i < MAX_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Launch memory transfers and kernel executions in parallel with streams
    int chunkSize = rows / MAX_STREAMS; // You can adjust this value
    for (int i = 0; i < MAX_STREAMS; i++)
    {
        int startRow = i * chunkSize;
        int endRow = (i == MAX_STREAMS - 1) ? rows : (i + 1) * chunkSize;
        void *chunkStart = (char *)deviceMatrix + startRow * cols * elementSize;
        int numRows = (i == MAX_STREAMS - 1) ? (rows - startRow) : chunkSize; // Rows for this stream

        // Asynchronously copy a chunk of data to the device
        cudaMemcpyAsync((char *)deviceMatrix + startRow * cols * elementSize,
                        (char *)matrix + startRow * cols * elementSize,
                        (endRow - startRow) * cols * elementSize,
                        cudaMemcpyHostToDevice, streams[i]);

        // Launch kernel after copying each chunk
        switch (type)
        {
        case INT:
            sortRowsKernelInt<<<dim3(numRows), block1D, sharedMemSize, streams[i]>>>((int *)chunkStart, numRows, cols);
            break;

        case FLOAT:
            sortRowsKernelFloat<<<dim3(numRows), block1D, sharedMemSize, streams[i]>>>((float *)chunkStart, numRows, cols);
            break;

        case DOUBLE:
            sortRowsKernelDouble<<<dim3(numRows), block1D, sharedMemSize, streams[i]>>>((double *)chunkStart, numRows, cols);
            break;
        }
    }

    // Synchronize all streams
    cudaDeviceSynchronize();
    int temp;
    switch (type)
    {
    case INT:
        transposeKernelInt<<<grid2D, block2D>>>((int *)deviceMatrix, (int *)output, rows, cols);
        break;
    case FLOAT:
        transposeKernelFloat<<<grid2D, block2D>>>((float *)deviceMatrix, (float *)output, rows, cols);
        break;
    case DOUBLE:
        transposeKernelDouble<<<grid2D, block2D>>>((double *)deviceMatrix, (double *)output, rows, cols);
        break;
    }
    temp = rows;
    rows = cols;
    cols = temp;
    cudaMemcpy(deviceMatrix, output, rows * cols * elementSize, cudaMemcpyDeviceToDevice);

    // Destroy the streams
    for (int i = 0; i < MAX_STREAMS; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    // Launch kernels for sorting
    while (!isMatrixSorted(deviceMatrix, rows, cols, type, foundUnsorted))
    {
        switch (type)
        {
        case INT:
            sortRowsKernelInt<<<grid1D, block1D, sharedMemSize>>>((int *)deviceMatrix, rows, cols);
            transposeKernelInt<<<grid2D, block2D>>>((int *)deviceMatrix, (int *)output, rows, cols);
            break;
        case FLOAT:
            sortRowsKernelFloat<<<grid1D, block1D, sharedMemSize>>>((float *)deviceMatrix, rows, cols);
            transposeKernelFloat<<<grid2D, block2D>>>((float *)deviceMatrix, (float *)output, rows, cols);

            break;
        case DOUBLE:
            sortRowsKernelDouble<<<grid1D, block1D, sharedMemSize>>>((double *)deviceMatrix, rows, cols);
            transposeKernelDouble<<<grid2D, block2D>>>((double *)deviceMatrix, (double *)output, rows, cols);
            break;
        }
        temp = rows;
        rows = cols;
        cols = temp;
        cudaMemcpy(deviceMatrix, output, rows * cols * elementSize, cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();

    // Copy result back to host (overlap already handled in streams)
    copyMatrixFromDeviceToHost(deviceMatrix, matrix, rows, cols, type);

    // Free device memory
    cudaFree(deviceMatrix);
    cudaFree(output);
    cudaFree(foundUnsorted);

    // Write the matrix to file
    writeMatrix(argv[2], matrix, rows, cols, type);

    return 0;
}

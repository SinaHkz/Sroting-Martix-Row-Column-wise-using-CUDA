#include "utilities.h"

#define BLOCKSIZE 32

// Function to read a matrix from "input.txt"
void* readMatrix(const char* filename, int* rows, int* cols, MatrixType* type) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    char typeStr[10];
    // Read matrix type and dimensions
    if (fscanf(file, "%s %d %d", typeStr, rows, cols) != 3) {
        fprintf(stderr, "Error reading matrix type and dimensions\n");
        fclose(file);
        return NULL;
    }

    if (strcmp(typeStr, "int") == 0) {
        *type = INT;
    } else if (strcmp(typeStr, "float") == 0) {
        *type = FLOAT;
    } else if (strcmp(typeStr, "double") == 0) {
        *type = DOUBLE;
    } else {
        fprintf(stderr, "Unsupported matrix type\n");
        fclose(file);
        return NULL;
    }

    size_t elementSize;
    if (*type == INT) {
        elementSize = sizeof(int);
    } else if (*type == FLOAT) {
        elementSize = sizeof(float);
    } else {
        elementSize = sizeof(double);
    }

    // Allocate memory for the matrix as a single contiguous block
    void* matrix = malloc(*rows * *cols * elementSize);
    if (matrix == NULL) {
        perror("Error allocating memory for matrix");
        fclose(file);
        return NULL;
    }

    // Read matrix elements into the contiguous memory block
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            size_t index = i * (*cols) + j;
            if (*type == INT) {
                if (fscanf(file, "%d", &((int*)matrix)[index]) != 1) {
                    fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                    free(matrix);
                    fclose(file);
                    return NULL;
                }
            } else if (*type == FLOAT) {
                if (fscanf(file, "%f", &((float*)matrix)[index]) != 1) {
                    fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                    free(matrix);
                    fclose(file);
                    return NULL;
                }
            } else if (*type == DOUBLE) {
                if (fscanf(file, "%lf", &((double*)matrix)[index]) != 1) {
                    fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                    free(matrix);
                    fclose(file);
                    return NULL;
                }
            }
        }
    }

    fclose(file);
    return matrix;
}

// Function to write a matrix to a file
void writeMatrix(const char* filename, void* matrix, int rows, int cols, MatrixType type) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file for writing");
        return;
    }

    // Write matrix type and dimensions
    const char* typeStr = (type == INT) ? "int" : (type == FLOAT) ? "float" : "double";
    fprintf(file, "%s %d %d\n", typeStr, rows, cols);

    // Write matrix elements from the contiguous memory block
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            size_t index = i * cols + j;
            if (type == INT) {
                fprintf(file, "%d ", ((int*)matrix)[index]);
            } else if (type == FLOAT) {
                fprintf(file, "%f ", ((float*)matrix)[index]);
            } else if (type == DOUBLE) {
                fprintf(file, "%lf ", ((double*)matrix)[index]);
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}


// Main function to check if the matrix is sorted for different types
bool isMatrixSorted(void *matrix, int rows, int cols, MatrixType type) {
    int *foundUnsorted;
    dim3 block(BLOCKSIZE);
    dim3 grid((rows + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    cudaMallocManaged(&foundUnsorted, sizeof(int));
    cudaMemset(foundUnsorted, 0, sizeof(int));

    // Choose the correct kernel based on the type
    switch (type) {
        case INT: {
            // Check for int type
            checkSortedRowWiseInt<<<grid, block>>>((int*)matrix, rows, cols, foundUnsorted);
            checkSortedColumnWiseInt<<<grid, block>>>((int*)matrix, rows, cols, foundUnsorted);
            break;
        }
        case FLOAT: {
            // Check for float type
            checkSortedRowWiseFloat<<<(rows + 255) / 256, 256>>>((float*)matrix, rows, cols, foundUnsorted);
            checkSortedColumnWiseFloat<<<(cols + 255) / 256, 256>>>((float*)matrix, rows, cols, foundUnsorted);
            break;
        }
        case DOUBLE: {
            // Check for double type
            checkSortedRowWiseDouble<<<(rows + 255) / 256, 256>>>((double*)matrix, rows, cols, foundUnsorted);
            checkSortedColumnWiseDouble<<<(cols + 255) / 256, 256>>>((double*)matrix, rows, cols, foundUnsorted);
            break;
        }
        default:
            printf("Invalid type\n");
            cudaFree(foundUnsorted);
            return false;
    }

    // Synchronize and retrieve the result
    cudaDeviceSynchronize();

    int unsorted;
    cudaMemcpy(&unsorted, foundUnsorted, sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(foundUnsorted);

    // Return true if sorted, false if unsorted
    return unsorted == 0;
}
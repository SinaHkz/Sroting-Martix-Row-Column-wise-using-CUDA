#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to read the matrix from a file
int** readMatrixFromFile(const char* filename, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }

    // Read the number of rows and columns from the file
    fscanf(file, "int %d %d", rows, cols);

    // Dynamically allocate memory for the matrix
    int** matrix = (int**)malloc(*rows * sizeof(int*));
    for (int i = 0; i < *rows; ++i) {
        matrix[i] = (int*)malloc(*cols * sizeof(int));
    }

    // Read the matrix values from the file
    for (int i = 0; i < *rows; ++i) {
        for (int j = 0; j < *cols; ++j) {
            fscanf(file, "%d", &matrix[i][j]);
        }
    }

    fclose(file);
    return matrix;
}

// Function to compare if two matrices are transposed versions of each other
int areMatricesTransposed(int** input, int** output, int inputRows, int inputCols, int outputRows, int outputCols) {
    // The matrix dimensions should be swapped for the transpose condition
    if (inputCols != outputRows || inputRows != outputCols) {
        return 0; // Matrices cannot be transposed if dimensions don't match
    }

    // Compare the values to check if the output matrix is the transpose of the input matrix
    for (int i = 0; i < inputRows; ++i) {
        for (int j = 0; j < inputCols; ++j) {
            if (input[i][j] != output[j][i]) {
                return 0; // The matrices are not transposed
            }
        }
    }

    return 1; // The matrices are transposed
}

int main() {
    int inputRows, inputCols, outputRows, outputCols;

    // Read the matrices from the input and output files
    int** inputMatrix = readMatrixFromFile("../data/input.txt", &inputRows, &inputCols);
    int** outputMatrix = readMatrixFromFile("../data/output.txt", &outputRows, &outputCols);

    if (inputMatrix == NULL || outputMatrix == NULL) {
        return 1; // Error reading files
    }

    // Check if the output matrix is the transpose of the input matrix
    if (areMatricesTransposed(inputMatrix, outputMatrix, inputRows, inputCols, outputRows, outputCols)) {
        printf("The output file is the transpose of the input file.\n");
    } else {
        printf("The output file is NOT the transpose of the input file.\n");
    }

    // Free the allocated memory
    for (int i = 0; i < inputRows; ++i) {
        free(inputMatrix[i]);
    }
    free(inputMatrix);

    for (int i = 0; i < outputRows; ++i) {
        free(outputMatrix[i]);
    }
    free(outputMatrix);

    return 0;
}

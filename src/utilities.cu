#include "utilities.h"

// Function to read a matrix from "input.txt"
void** readMatrix(const char* filename, int* rows, int* cols, MatrixType* type) {
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

    // Allocate memory for the matrix
    void** matrix = (void** )malloc(*rows * sizeof(void*));
    if (matrix == NULL) {
        perror("Error allocating memory for matrix rows");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < *rows; i++) {
        matrix[i] = malloc(*cols * elementSize);
        if (matrix[i] == NULL) {
            perror("Error allocating memory for matrix columns");
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            fclose(file);
            return NULL;
        }
    }

    // Read matrix elements
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (*type == INT) {
                if (fscanf(file, "%d", &((int*)matrix[i])[j]) != 1) {
                    fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                    goto cleanup;
                }
            } else if (*type == FLOAT) {
                if (fscanf(file, "%f", &((float*)matrix[i])[j]) != 1) {
                    fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                    goto cleanup;
                }
            } else if (*type == DOUBLE) {
                if (fscanf(file, "%lf", &((double*)matrix[i])[j]) != 1) {
                    fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                    goto cleanup;
                }
            }
        }
    }

    fclose(file);
    return matrix;

cleanup:
    for (int i = 0; i < *rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
    fclose(file);
    return NULL;
}

// Function to write a matrix to a file
void writeMatrix(const char* filename, void** matrix, int rows, int cols, MatrixType type) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file for writing");
        return;
    }

    // Write matrix type and dimensions
    const char* typeStr = (type == INT) ? "int" : (type == FLOAT) ? "float" : "double";
    fprintf(file, "%s %d %d\n", typeStr, rows, cols);

    // Write matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (type == INT) {
                fprintf(file, "%d ", ((int*)matrix[i])[j]);
            } else if (type == FLOAT) {
                fprintf(file, "%f ", ((float*)matrix[i])[j]);
            } else if (type == DOUBLE) {
                fprintf(file, "%lf ", ((double*)matrix[i])[j]);
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}


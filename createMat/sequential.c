#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 10

// Function to read a matrix from a file
void *readMatrix(const char *filename, int *rows, int *cols)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        return NULL;
    }

    char typeStr[10];
    if (fscanf(file, "%s %d %d", typeStr, rows, cols) != 3)
    {
        fprintf(stderr, "Error reading matrix type and dimensions\n");
        fclose(file);
        return NULL;
    }

    size_t elementSize = sizeof(int);

    void *matrix = malloc(*rows * *cols * elementSize);
    if (matrix == NULL)
    {
        perror("Error allocating memory for matrix");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < *rows; i++)
    {
        for (int j = 0; j < *cols; j++)
        {
            size_t index = i * (*cols) + j;
            if (fscanf(file, "%d", &((int *)matrix)[index]) != 1)
            {
                fprintf(stderr, "Error reading matrix element at (%d, %d)\n", i, j);
                free(matrix);
                fclose(file);
                return NULL;
            }
        }
    }

    fclose(file);
    return matrix;
}

// Function to write a matrix to a file
void writeMatrix(const char *filename, void *matrix, int rows, int cols)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Error opening file for writing");
        return;
    }

    const char *typeStr = "int";
    fprintf(file, "%s %d %d\n", typeStr, rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            size_t index = i * cols + j;
            fprintf(file, "%d ", ((int *)matrix)[index]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Function to sequentially sort the matrix row-wise and column-wise
int matrixSortSequential(int *mat, int n)
{
    int count = 0;
    int sorted = 0;
    while (!sorted)
    {
        sorted = 1;
        // Row-wise sort
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n - 1; j++)
            {
                if (mat[i * n + j] > mat[i * n + j + 1])
                {
                    int temp = mat[i * n + j];
                    mat[i * n + j] = mat[i * n + j + 1];
                    mat[i * n + j + 1] = temp;
                    sorted = 0;
                }
            }
        }

        // Column-wise sort
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n - 1; i++)
            {
                if (mat[i * n + j] > mat[(i + 1) * n + j])
                {
                    int temp = mat[i * n + j];
                    mat[i * n + j] = mat[(i + 1) * n + j];
                    mat[(i + 1) * n + j] = temp;
                    sorted = 0;
                }
            }
        }
        count++;
    }
    return count;
}

// Function to print the matrix
void printMat(int *mat, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ", mat[i * n + j]);
        }
        printf("\n");
    }
}

// Driver program to test the sorting function
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    int rows, cols;
    int *mat = (int *)readMatrix(argv[1], &rows, &cols);
    if (mat == NULL)
    {
        return 1;
    }

    // printf("Original Matrix:\n");
    // printMat(mat, rows);

    // Sorting the matrix until fully sorted
    int sortCount = matrixSortSequential(mat, rows);

    // printf("\nMatrix After Sorting (%d iterations):\n", sortCount);
    // printMat(mat, rows);

    // Writing the sorted matrix to the output file
    writeMatrix(argv[2], mat, rows, cols);

    free(mat);
    return 0;
}

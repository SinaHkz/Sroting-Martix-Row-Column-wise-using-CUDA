#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to generate a random integer matrix
void generateRandomMatrix(int rows, int cols, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write the matrix dimensions to the file
    fprintf(file, "int %d %d\n", rows, cols);

    // Seed the random number generator
    srand(time(NULL));

    // Generate and write the random matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int randomValue = rand() % 100; // Random values between 0 and 99
            fprintf(file, "%d ", randomValue);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Matrix saved to %s\n", filename);
}

int main() {
    int rows, cols;
    const char *filename = "../data/input.txt";

    printf("Enter the number of rows: ");
    scanf("%d", &rows);
    printf("Enter the number of columns: ");
    scanf("%d", &cols);

    generateRandomMatrix(rows, cols, filename);

    return 0;
}

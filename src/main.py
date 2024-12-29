import random
import subprocess
import os

def generate_square_matrix(matrix_type, size):
    if matrix_type == 'int':
        return [[random.randint(1, 100) for _ in range(size)] for _ in range(size)]
    elif matrix_type == 'double':
        return [[round(random.uniform(1, 10), 2) for _ in range(size)] for _ in range(size)]
    else:
        raise ValueError("Unsupported matrix type")

def save_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        file.write(f"{len(matrix)} {len(matrix[0])}\n")
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")


def main():
    matrix_sizes = [1024, 2048, 4096, 8192, 16384]
    
    matrix_type = input("Do you want an 'int' or 'double' matrix? ").strip().lower()
    if matrix_type not in ['int', 'double']:
        raise ValueError("Invalid input. Please enter 'int' or 'double'.")

    input_dir = "input_matrices"
    os.makedirs(input_dir, exist_ok=True)

    for size in matrix_sizes:
        print(f"\nProcessing matrix of size {size}x{size}...\n")

        matrix = generate_square_matrix(matrix_type, size)

        matrix_file = f'{input_dir}/generated_square_matrix_{matrix_type}_{size}.txt'
        save_matrix_to_file(matrix, matrix_file)
        print(f"Matrix saved to '{matrix_file}'.")

if __name__ == "__main__":
    main()

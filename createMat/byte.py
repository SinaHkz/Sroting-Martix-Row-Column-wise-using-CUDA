def compare_files(file1, file2):
    try:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            byte_position = 0
            line_number = 1
            differences_count = 0  # Initialize counter for differences

            while True:
                byte1 = f1.read(1)
                byte2 = f2.read(1)

                # If both files have ended, stop comparison
                if not byte1 and not byte2:
                    if differences_count == 0:
                        print("The files are identical.")
                    break  # End the loop if both files have ended

                # If one file has ended but the other hasn't, they are different
                if not byte1 or not byte2:
                    differences_count += 1
                    print(f"Files differ at byte {byte_position}, line {line_number}")
                    print(f"Byte in {file1}: {'End of file'}")
                    print(f"Byte in {file2}: {'End of file'}")
                    break

                if byte1 != byte2:
                    differences_count += 1  # Increment the counter for differences
                    # Print the byte position, line number, and the differing byte values
                    print(f"Files differ at byte {byte_position}, line {line_number}")
                    print(f"Byte in {file1}: {byte1[0]} (0x{byte1[0]:02x})")
                    print(f"Byte in {file2}: {byte2[0]} (0x{byte2[0]:02x})")
                
                # Track byte position and line number
                byte_position += 1
                if byte1 == b'\n':
                    line_number += 1

            # After the loop, print the total number of differences if any
            print(f"\nTotal number of differences: {differences_count}")

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Example usage with the specified file paths:
file1 = '../data/input.txt'  # Replace with your file path
file2 = '../data/output.txt'  # Replace with your file path
compare_files(file1, file2)

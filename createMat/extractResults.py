import matplotlib.pyplot as plt # type: ignore

# Data for matrix sizes and memory/time values
matrix_sizes = [1024, 2048, 4096, 8192]
host_to_device_memory = [336, 344, 352, 1875.00]
device_to_host_memory = [336, 344, 352, 1875.00]
memory_thrashes = [84, 84, 84, 192]

gpu_time = [200.89 / 1000, 904.31 / 1000, 6.75567, 132.218]
api_time = [179.34 / 1000, 893.49 / 1000, 6.78523, 132.505]
total_time = [gpu_time[i] + api_time[i] for i in range(len(gpu_time))]

# Plot for Memory Data
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, host_to_device_memory, label='Host to Device (KB/MB)', marker='o')
plt.plot(matrix_sizes, device_to_host_memory, label='Device to Host (KB/MB)', marker='o')
plt.plot(matrix_sizes, memory_thrashes, label='Memory Thrashes (KB)', marker='o')

plt.xlabel('Matrix Size')
plt.ylabel('Memory (KB / MB)')
plt.title('Memory Copied for Different Matrix Sizes')
plt.legend()
plt.grid(True)

# Save memory plot
plt.tight_layout()
plt.savefig('./memory_plot_updated.png')

# Plot for Time Data
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, gpu_time, label='GPU Time (s)', marker='o')
plt.plot(matrix_sizes, api_time, label='API Time (s)', marker='o')
plt.plot(matrix_sizes, total_time, label='Total Time (s)', marker='o')

plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title('Time Taken for Different Matrix Sizes')
plt.legend()
plt.grid(True)

# Save time plot
plt.tight_layout()
plt.savefig('./time_plot_updated.png')

# Output the plot file paths
memory_plot_path = './memory_plot_updated.png'
time_plot_path = './time_plot_updated.png'

memory_plot_path, time_plot_path

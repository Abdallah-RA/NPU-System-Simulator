import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "npu_statistics.csv"
try:
    data = pd.read_csv(csv_file, on_bad_lines="skip")  # Skip any problematic lines
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# Ensure the data types are correct
data = data.dropna()  # Drop rows with missing values
data["Vector Length"] = data["Vector Length"].astype(int)
data["Workloads"] = data["Workloads"].astype(int)
data["Execution Time (seconds)"] = data["Execution Time (seconds)"].astype(float)
data["Throughput (tasks/sec)"] = data["Throughput (tasks/sec)"].astype(float)
data["Total Cycles"] = data["Total Cycles"].astype(float)

# Print statistics for all operations as a table
print("Statistics for all operations:")
statistics_table = data.groupby(["Operation", "Vector Length", "Workloads"]).agg(
    {
        "Execution Time (seconds)": "mean",
        "Throughput (tasks/sec)": "mean",
        "Total Cycles": "mean",
    }
).reset_index()

# Display the statistics table
print(statistics_table)

# Filter data for Vector Length = 1024
vector_length = 1024
filtered_data = data[data["Vector Length"] == vector_length]

# Separate data for Classical and Karatsuba
classical_data = filtered_data[filtered_data["Operation"] == "Classical"]
karatsuba_data = filtered_data[filtered_data["Operation"] == "Karatsuba"]

# Plot: Classical vs Karatsuba (Vector Length = 1024) - Workloads vs Execution Time
plt.figure(figsize=(10, 6))

# Plot Classical data
plt.plot(
    classical_data["Workloads"],
    classical_data["Execution Time (seconds)"],
    marker="o",
    label="Classical"
)

# Plot Karatsuba data
plt.plot(
    karatsuba_data["Workloads"],
    karatsuba_data["Execution Time (seconds)"],
    marker="o",
    label="Karatsuba"
)

# Add title, labels, and legend
plt.title(f"Classical vs Karatsuba (Vector Length = {vector_length}): Workloads vs Execution Time")
plt.xlabel("Number of Workloads")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Separate data for VPU
vpu_data = data[data["Operation"] == "VPU"]

# Plot: VPU - Workloads vs Execution Time
plt.figure(figsize=(10, 6))
for vector_length in vpu_data["Vector Length"].unique():
    subset = vpu_data[vpu_data["Vector Length"] == vector_length]
    plt.plot(
        subset["Workloads"],
        subset["Execution Time (seconds)"],
        marker="o",
        label=f"Vector Length={vector_length}"
    )
plt.title("VPU: Workloads vs Execution Time")
plt.xlabel("Number of Workloads")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.grid(True)
plt.show()

